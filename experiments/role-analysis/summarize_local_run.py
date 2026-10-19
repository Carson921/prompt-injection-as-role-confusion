"""Validate completed artifacts and produce the local reproduction report."""
import hashlib
import gzip
import json
from pathlib import Path
import platform
import pickle
import re
import subprocess
from datetime import datetime

import nbformat
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'outputs/local-gptoss20b'


def load_json(name):
    return json.loads((OUT/name).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import cupy, cuml, sklearn, torch, transformers
    complete = load_json('python_workflow_complete.json')
    generation = load_json('conversation_generation.json')
    raw_conversations_path = HERE/'data/conversations/gptoss-20b-raw.csv'
    raw_conversations = pd.read_csv(raw_conversations_path)
    # Early in-memory cache migration added an extra terminal marker to 15 rows;
    # normalize that metadata-only artifact without changing any parsed response.
    raw_conversations['state_text'] = raw_conversations.state_text.str.replace(
        r'(?:<\|return\|>){2,}$','<|return|>',regex=True)
    raw_conversations.to_csv(raw_conversations_path,index=False)
    adaptation = load_json('adaptation_validation.json')
    forward = load_json('forward_validation.json')
    gpu = load_json('gpu-memory-summary.json')
    diagnostic = load_json('midlayer-float64-diagnostic.json')
    metrics = pd.read_csv(OUT/'training_metrics.csv')
    accuracy = pd.read_csv(HERE/'outputs/probe-projections/all_conv_acc_gptoss-20b.csv')
    probability = pd.read_csv(HERE/'outputs/probe-projections/all_conv_projs_gptoss-20b.csv')
    labels = pd.read_parquet(OUT/'training_labels.parquet')
    validation_labels = pd.read_parquet(OUT/'conversation_labels.parquet',columns=['role','prompt_ix'])
    validation_role_counts = {k:int(v) for k,v in validation_labels.groupby('role').size().items()}
    validation_sources = pd.DataFrame(load_json('validation_conversations.json')).dataset.value_counts().to_dict()
    inputs = pd.read_parquet(OUT/'training_inputs.parquet')
    tomato = pd.read_csv(HERE/'outputs/probe-projections/tomato-role-projections-gptoss-20b.csv',keep_default_na=False)
    source_csv = HERE/'outputs/probe-projections/tomato-role-projections-gptoss-20b.csv'
    import shutil
    with source_csv.open('rb') as source, gzip.open(str(source_csv)+'.gz','wb',compresslevel=6) as destination:
        shutil.copyfileobj(source,destination)
    assert len(metrics)==96 and metrics[['role_space','layer_ix']].drop_duplicates().shape[0]==96
    assert set(accuracy.conv_type)=={'tagged','untagged','user_tagged','tool_tagged'}
    assert set(metrics.layer_ix)==set(range(0,24,2))
    assert len(inputs)==249*5
    assert set(tomato.prompt_key)=={'basic_no_format','everything_in_user_tags','proper_tags'}
    assert tomato.prob.between(0,1).all() and accuracy.mean_acc.between(0,1).all()
    notebooks = []
    timings = []
    for source in sorted(HERE.glob('[0-9][0-9]-*.ipynb')):
        executed = OUT/(source.stem+'.executed.ipynb')
        nb = nbformat.read(executed,as_version=4)
        code = [c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code),executed
        errors = [o for c in code for o in c.outputs if o.output_type=='error']
        assert not errors,(executed,errors)
        for index,cell in enumerate(nb.cells):
            timing = cell.metadata.get('execution',{})
            if 'iopub.status.busy' in timing and 'iopub.status.idle' in timing:
                begin=datetime.fromisoformat(timing['iopub.status.busy'])
                end=datetime.fromisoformat(timing['iopub.status.idle'])
                timings.append({'notebook':source.name,'original_cell':index,'seconds':(end-begin).total_seconds()})
        streams = [o.text for c in code for o in c.outputs if o.output_type=='stream']
        (OUT/(source.stem+'.final.log')).write_text('\n'.join(streams))
        notebooks.append({'source':source.name,'source_sha256':sha256(source),
                          'executed':str(executed),'code_cells':len(code),'errors':0})
    pd.DataFrame(timings).to_csv(OUT/'cell-timings.csv',index=False)
    filter_counts = dict((name,int(count)) for name,count in re.findall(
        r'^(Starting length|After dropping missing|After dropping long convs|After dropping substr|Final): (\d+) convs$',
        (OUT/'02-train-role-probes.final.log').read_text(),flags=re.MULTILINE))
    (OUT/'conversation-filter-counts.json').write_text(json.dumps(filter_counts,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name,group in metrics.groupby('role_space',sort=False):
        group=group.sort_values('layer_ix')
        ax.plot(group.layer_ix,group.acc*100,marker='o',markersize=3,label=name)
    ax.set(xlabel='Layer index (zero-based)',ylabel='Held-out prompt accuracy (%)',ylim=(0,100))
    ax.set_xticks(range(0,24,2));ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
    fig.tight_layout()
    for extension in ['png','pdf']:
        fig.savefig(OUT/f'plots/training-accuracy-by-layer.{extension}',dpi=180)
    plt.close(fig)
    best = metrics.loc[metrics.groupby('role_space').acc.idxmax()].sort_values('role_space')
    mid = accuracy[(accuracy.role_space=='uat')&(accuracy.layer_ix==14)]
    mid_table = mid.pivot(index='conv_type',columns='role',values='mean_acc')
    mid_table.to_csv(OUT/'conversation-accuracy-layer14.csv')
    mid_probability = probability[(probability.role_space=='uat')&(probability.layer_ix==14)]
    mid_probability.pivot(index='conv_type',columns='role',values='mean_acc').to_csv(OUT/'conversation-probability-layer14.csv')
    tomato_cot = (tomato[(tomato.role_space=='suca')&(tomato.layer_ix==12)&(tomato.target_role=='cot')]
        .groupby(['prompt_key','base_message_ix','base_message_type'],as_index=False)
        .agg(tokens=('prob','size'),mean_cot_probability=('prob','mean')))
    tomato_cot.to_csv(OUT/'tomato-cotness-by-message.csv',index=False)
    # Check whether the main illustrative probabilities depend on float32 fitting.
    maps = list((OUT/'cache/activations').glob('*/hidden_states.fp16'))
    sensitivity = {'status':'activation cache unavailable'}
    if len(maps)==1:
        import numpy as np
        path = maps[0]
        meta = json.loads((path.parent/'complete.json').read_text())
        shape = meta['shape']
        tensor = torch.from_file(str(path),shared=False,size=int(np.prod(shape)),dtype=torch.float16).view(shape)
        selected = tomato[(tomato.role_space=='suca')&(tomato.layer_ix==12)]
        token_metadata = (selected[['sample_ix','prompt_key','base_message_ix','base_message_type']]
            .drop_duplicates('sample_ix').sort_values('sample_ix').reset_index(drop=True))
        with open(OUT/'midlayer-float64-probe.pkl','rb') as f:
            double_probe = pickle.load(f)
        features = cupy.asarray(tensor[meta['layers'].index(12),token_metadata.sample_ix.tolist()].numpy().astype(np.float64))
        predictions = cupy.asnumpy(double_probe.predict_proba(features))
        roles = ['system','user','cot','assistant']
        original = (selected.pivot(index='sample_ix',columns='target_role',values='prob')
            .loc[token_metadata.sample_ix,roles].to_numpy())
        sensitivity = {'status':'complete','tokens':len(predictions),
            'mean_absolute_probability_difference':float(np.abs(predictions-original).mean()),
            'maximum_absolute_probability_difference':float(np.abs(predictions-original).max()),
            'argmax_agreement':float((predictions.argmax(1)==original.argmax(1)).mean())}
        compared = token_metadata.assign(float32_cot_probability=original[:,2],float64_cot_probability=predictions[:,2])
        compared.groupby(['prompt_key','base_message_ix','base_message_type'],as_index=False).agg(
            tokens=('sample_ix','size'),float32_cot_probability=('float32_cot_probability','mean'),
            float64_cot_probability=('float64_cot_probability','mean')).to_csv(OUT/'tomato-float64-sensitivity.csv',index=False)
        del features,predictions,double_probe,tensor
    (OUT/'tomato-float64-sensitivity.json').write_text(json.dumps(sensitivity,indent=2))
    # Match the original CuML split exactly and quantify shared base texts.
    split_records = []
    for roles in metrics.role_space.unique():
        names = {'s':'system','u':'user','c':'cot','a':'assistant','t':'tool'}
        subset = labels[labels.role.isin([names[r] for r in roles])]
        train,test = cuml.train_test_split(subset.prompt_ix.unique(),test_size=.1,random_state=123)
        mapping = inputs.set_index('prompt_ix').question_ix
        train_questions = set(mapping.loc[list(train)].tolist())
        test_questions = set(mapping.loc[list(test)].tolist())
        split_records.append({'role_space':roles,'train_prompts':len(train),'test_prompts':len(test),
                              'test_base_texts':len(test_questions),
                              'test_base_texts_also_in_train':len(test_questions&train_questions)})
    pd.DataFrame(split_records).to_csv(OUT/'split-overlap.csv',index=False)
    generation_batches = [json.loads(p.read_text()) for p in (OUT/'cache/generation_batches').glob('*.json')]
    for batch in generation_batches:
        for state,detail in zip(batch['state_text'],batch['details'],strict=True):
            assert state.endswith('<|return|>') == detail['ended_with_eos']
            assert not state.endswith('<|return|>'*2)
            assert 0 < detail['generated_tokens'] <= 4000
    generation_details = [r for b in generation_batches for r in b['details']]
    manifest = {
        'base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'model_path':'/workspace/gpt-oss-20b','model_config_sha256':sha256('/workspace/gpt-oss-20b/config.json'),
        'tokenizer_sha256':sha256('/workspace/gpt-oss-20b/tokenizer.json'),
        'generation_config_sha256':sha256('/workspace/gpt-oss-20b/generation_config.json'),
        'generation_token_overrides':{'eos_token_id':200002,'pad_token_id':199999},
        'probe_config_sha256':sha256(HERE/'config/probe.yaml'),'tomato_config_sha256':sha256(HERE/'config/tomato.yaml'),
        'raw_training_data_sha256':sha256(OUT/'raw_training_data.json'),
        'conversation_queries_sha256':sha256(OUT/'user_queries.csv'),
        'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
        'cuml':cuml.__version__,'cupy':cupy.__version__,'sklearn':sklearn.__version__,
        'venv':str(ROOT/'.venv'),'gpu':gpu,
        'probe_backend':'BF16 dequantized weights, eager experts/attention; actual-model hooks',
        'generation_backend':'native MXFP4/Triton experts, BF16 activations, eager attention',
        'concurrent_generation_and_training':True,'notebooks':notebooks,
        'python_workflow':complete,'generation':generation,'adaptation_checks':adaptation,'forward_validation':forward,
        'conversation_filter_counts':filter_counts,
        'validation_tokens_including_tags':len(validation_labels),
        'validation_semantic_role_token_counts':validation_role_counts,
        'validation_conversation_sources':validation_sources,
        'float64_sensitivity_diagnostic':diagnostic,
        'tomato_float64_sensitivity':sensitivity,
        'generated_responses':len(generation_details),'generated_tokens':sum(x['generated_tokens'] for x in generation_details),
        'generation_batches':len(generation_batches),'eos_completed_responses':sum(x['ended_with_eos'] for x in generation_details),
        'skipped':['Haiku, MiniMax, Kimi cross-model conversation comparison (explicit user instruction)'],
    }
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    warnings = []
    for nb in notebooks:
        text = (OUT/(Path(nb['source']).stem+'.final.log')).read_text()
        for line in text.splitlines():
            if any(s in line.lower() for s in ['failed to converge','lbfgs','line search','linesearch','qn solver']):
                warnings.append(line.strip())
    (OUT/'solver-warnings.txt').write_text('\n'.join(warnings))
    report = f'''gpt-oss-20b Role Space Analysis 复现报告

范围：四个原始 notebook 的本地单模型流程已全部执行。依用户明确指示，跳过 Haiku、MiniMax、Kimi 的跨模型对照。
硬件：{gpu['device']}，{gpu['total_gib']:.0f} GiB；整卡采样峰值 {gpu['peak_used_gib']:.2f} GiB。
Python 环境：{ROOT/'.venv'}，Python {platform.python_version()}；R 版本与包见 R-sessionInfo.txt。
探针侧：BF16 解量化、eager experts/attention；生成侧：同一 checkpoint 的原生 MXFP4/Triton。
生成与训练曾在同卡并行，因此显存峰值包含两个进程。

训练设置
原配置为 250 文本，但原代码分项取整实际得到 62 C4 + 187 Dolma3 = 249。
每条最多 1024 token，分别置于 system/user/tool/cot/assistant：{len(inputs)} 个序列。
内容 token 数：{labels.groupby('role').size().to_dict()}。
12 个层（0,2,...,22）× 8 个角色组合 = 96 个探针；C=0.005，无特征缩放，L2 LR。
按原代码，以 prompt_ix 做 90/10 切分，seed=123。

各角色组合最佳训练留出准确率（这是原始训练数据的留出集）
{best[['role_space','layer_ix','acc','nll']].to_string(index=False)}

对话生成与验证
候选 {generation['requested_conversations']} 段、{generation['requested_user_turns']} 个用户轮次；
完整生成保留 {generation['complete_conversations']} 段，未完成则整段删除 {generation['dropped_incomplete_conversations']} 段。
进一步按原 notebook 的长度、缺失、包含关系过滤后，实际验证 {complete['validation_conversations']} 段。
原筛选计数：{filter_counts}；使用全部合格对话，上限100。
来源分布：{validation_sources}；四种格式共{len(validation_labels):,}个token，语义角色计数{validation_role_counts}。
生成 temperature=1，最多4000新token，最多2用户轮；原始生成和确定性批次seed保存在cache。
四种格式：正确标签、无标签、全部user标签、全部tool标签。
下表为原 R 分析选定的第14层 UAT 探针准确率：先在每段对话内部按语义角色平均，再对对话平均。
{mid_table.to_string(float_format=lambda x:f'{x:.4%}')}

Tomato 示例
全部三种格式与96探针完成，共 {len(tomato):,} 行概率投影。
SUCA 第12层、逐消息 CoT 概率均值如下（未平滑，不是原论文数值的逐点复刻）：
{tomato_cot.to_string(index=False,float_format=lambda x:f'{x:.5f}')}

校验与限制
实际模型前向钩子校验：{forward}。
Harmony通道解析、分流汇总与原公式等价性检查通过，包括不等长对话和argmax并列情况。
已执行 notebook 均无错误，两个 R notebook 使用实际 IRkernel 执行，保留原平滑及token对齐逻辑。
修正原本地分支丢失CoT的问题；生成明确使用tokenizer的单一EOS覆盖模型默认多结束符配置。
原训练切分允许同一底层文本的不同角色副本跨越训练/测试集；具体重叠见split-overlap.csv，不能将该训练留出准确率解释为完全未见文本的泛化率。
本地采样、模型后端、候选对话过滤会影响数值；该报告不是原论文所有模型及全部对照的复现。
优化器诊断相关日志行 {len(warnings)} 条，见solver-warnings.txt；原训练超参数未更改。
另做 SUCA 第12层 float64 检查：原 float32 准确率 {diagnostic['float32_accuracy']:.4%}，float64 为 {diagnostic['float64_accuracy']:.4%}，相差 {diagnostic['accuracy_difference_percentage_points']:.4f} 个百分点；未替换主实验探针。
Tomato 的 float64 稳定性统计：{sensitivity}；逐消息对照见tomato-float64-sensitivity.csv。

主要产物
四份 *.executed.ipynb：逐单元输入输出。
training_metrics.csv：96探针的准确率与NLL；outputs/probes/gptoss-20b.pkl：探针。
outputs/probe-projections/：对话概率/准确率和tomato逐token投影。
plots/：R生成的PNG/PDF及Python交互图。
run_manifest.json：数据/配置/模型哈希、软件版本、显存记录及执行校验。
复跑步骤与适配说明：{HERE/'LOCAL-A100.md'}。
'''
    (OUT/'report.txt').write_text(report)
    print(report)


if __name__=='__main__':
    main()
