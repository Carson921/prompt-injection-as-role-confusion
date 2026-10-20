"""Validate selected feature probes and report paired document-bootstrap results."""
import json
from pathlib import Path
import pickle

import search_feature_variants as run


def model_path(protocol,name,feature,C):
    return run.OUT/'models'/f'final-{protocol}-{name}-{feature}-{C:g}.pkl'


def main():
    import cupy as cp
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.probes import run_projections
    selected=json.loads((run.OUT/'selection.json').read_text())
    inputs=pd.read_parquet(run.SOURCE/'training_inputs.parquet').set_index('prompt_ix')
    labels=pd.read_parquet(run.SOURCE/'training_labels.parquet')
    n=json.loads((run.SOURCE/'manifest.json').read_text())['tokens']
    original=pd.read_csv(run.SOURCE/'training_metrics.csv')
    records=[]
    consumers=[]
    selected_models=[]
    doc_results=[]
    for protocol in run.engine.PROTOCOLS:
        for name in run.engine.ROLE_SPACES:
            choice=selected[protocol][name]
            _,test=run.engine.datasets(name,protocol,'final')
            test=test.assign(question_ix=test.prompt_ix.map(inputs.question_ix))
            predictions={}
            models={}
            for label,feature,C in [('baseline','norm-20',.005),('selected',choice['feature'],choice['C'])]:
                with model_path(protocol,name,feature,C).open('rb') as f:probe=pickle.load(f)
                assert probe['feature_spec']==run.FEATURES[feature]
                x=run.feature_array(feature,test)
                probabilities=probe['probe'].predict_proba(x)
                assert bool(cp.isfinite(probabilities).all()) and bool(cp.allclose(probabilities.sum(axis=1),1,atol=2e-6))
                pred=cp.asnumpy(probabilities.argmax(axis=1))
                truth=test.role.map(probe['roles_map']).to_numpy()
                correct=pred==truth
                assert abs(float(correct.mean())-probe['acc'])<1e-4
                predictions[label]=correct
                models[label]=probe
                del x,probabilities
                run.engine.cleanup()
            grouped=test.assign(baseline=predictions['baseline'],selected=predictions['selected']).groupby('question_ix').agg(
                tokens=('sample_ix','size'),baseline_correct=('baseline','sum'),selected_correct=('selected','sum')).reset_index()
            doc_results.append(grouped.assign(protocol=protocol,role_space=name))
            rng=np.random.RandomState(20261020)
            bootstrap=rng.randint(0,len(grouped),size=(2000,len(grouped)))
            differences=grouped.selected_correct.to_numpy()-grouped.baseline_correct.to_numpy()
            boot_gain=differences[bootstrap].sum(axis=1)/grouped.tokens.to_numpy()[bootstrap].sum(axis=1)*100
            lo,hi=map(float,np.quantile(boot_gain,[.025,.975]))
            before,after=models['baseline']['tuning']['evaluation'],models['selected']['tuning']['evaluation']
            records.append(dict(protocol=protocol,role_space=name,selected_feature=choice['feature'],C=choice['C'],
                                baseline_accuracy=before['accuracy'],selected_accuracy=after['accuracy'],
                                gain_percentage_points=100*(after['accuracy']-before['accuracy']),
                                gain_ci95_lower=lo,gain_ci95_upper=hi,test_documents=len(grouped),
                                baseline_cot_recall=before['recall']['cot'],selected_cot_recall=after['recall']['cot'],
                                baseline_system_recall=before['recall']['system'],selected_system_recall=after['recall']['system'],
                                original_best_accuracy=float(original[original.role_space==name].acc.max())))
            probe=models['selected']
            train_ids,test_ids=set(probe['train_prompt_ids']),set(probe['test_prompt_ids'])
            assert not train_ids&test_ids
            train_docs=set(inputs.loc[list(train_ids)].question_ix)
            test_docs=set(inputs.loc[list(test_ids)].question_ix)
            if protocol=='document':assert not train_docs&test_docs
            sample=test.groupby('role',group_keys=False).head(32).sort_values('sample_ix')
            parts=[]
            for spec in probe['feature_spec']:
                mmap=torch.from_file(str(run.feature_path(spec)),shared=False,size=n*2880,dtype=torch.float16).view(-1,2880)
                parts.append(mmap[sample.sample_ix.tolist()].clone())
            tensor=torch.cat(parts,dim=1)
            sample=sample.assign(sample_ix=range(len(sample)))
            got=run_projections(sample,tensor,probe).pivot(index='sample_ix',columns='target_role',values='prob')[probe['role_space']].to_numpy()
            direct=cp.asnumpy(probe['probe'].predict_proba(cp.asarray(tensor.float().numpy())))
            error=float(np.abs(got-direct).max())
            assert error<1e-4 and np.allclose(got.sum(axis=1),1,atol=2e-6)
            consumers.append(dict(protocol=protocol,role_space=name,feature=choice['feature'],
                                  rows=len(sample),dimensions=tensor.shape[1],max_probability_difference=error,
                                  train_test_prompt_disjoint=True,train_test_document_disjoint=not bool(train_docs&test_docs)))
            selected_models.append(probe)
            del models,parts,tensor,probe
            run.engine.cleanup()
    table=pd.DataFrame(records)
    table.to_csv(run.OUT/'comparison.csv',index=False)
    pd.concat(doc_results,ignore_index=True).to_csv(run.OUT/'test-document-counts.csv',index=False)
    with (run.OUT/'selected-probes.pkl').open('wb') as f:pickle.dump(selected_models,f)
    run.atomic_json(run.OUT/'consumer-validation.json',dict(all_passed=True,models=consumers))
    metrics=pd.read_csv(run.OUT/'metrics.csv')
    screen=metrics[metrics.stage=='screen'].pivot_table(index='feature',columns=['protocol','role_space'],values='accuracy',aggfunc='max')
    fig,ax=plt.subplots(figsize=(8,10))
    im=ax.imshow(screen.to_numpy(),vmin=.5,vmax=1,aspect='auto',cmap='viridis')
    ax.set(yticks=range(len(screen)),yticklabels=screen.index,xticks=range(4),
           xticklabels=[f'{a}\n{b}' for a,b in screen.columns],title='Screening validation accuracy (training cap: 256 tokens/prompt)')
    for i in range(len(screen)):
        for j in range(4):ax.text(j,i,f'{screen.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if screen.iloc[i,j]<.8 else 'black')
    fig.colorbar(im,ax=ax,label='Validation accuracy')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(run.OUT/f'feature-validation.{suffix}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,protocol in zip(axes,run.engine.PROTOCOLS):
        group=table[table.protocol==protocol]
        x=np.arange(len(group))
        a=ax.bar(x-.18,group.baseline_accuracy,width=.36,label='norm-20 baseline')
        b=ax.bar(x+.18,group.selected_accuracy,width=.36,label='Validation-selected features')
        ax.bar_label(a,labels=[f'{v:.2%}' for v in group.baseline_accuracy],fontsize=8,padding=3)
        ax.bar_label(b,labels=[f'{v:.2%}' for v in group.selected_accuracy],fontsize=8,padding=3)
        ax.set(xticks=x,xticklabels=group.role_space,ylim=(0,1),ylabel='Held-out token accuracy',title=protocol)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(run.OUT/f'test-comparison.{suffix}',dpi=180)
    plt.close(fig)
    lines=[
        'GPT-OSS-120B feature-location and multi-layer linear probe experiment','',
        '语言模型保持冻结的MXFP4权重。固定249篇原文、五种角色（含system）及原文分组划分均保留。',
        '新增特征：第16/20/24层注意力后的归一化前残差(mid)、MLP后层输出残差(out)，以及第18/22层归一化输出(norm)。',
        '8份新激活缓存共约31GiB；全部726,109个token的控制norm-16与原缓存逐值相同。',
        '新hook不改变logits，流水线与顺序执行的全部新抓取点逐值相同。',
        '共22种单层/双层/三层特征，均使用标准化及线性逻辑回归；没有用非线性分类器替代线性探针。',
        '筛选阶段训练token每个prompt均匀抽取至多256个，验证token不抽样；全部176个配置使用同一规则。',
        '按内部验证集选取每组前三种特征，加入norm-16/20控制，再用全部训练token及两个C值复核。',
        '最终根据完整训练token的验证accuracy、NLL选择，再用训练+验证原文重新拟合并评估测试集。',
        'legacy与document是不同评估协议，不能直接比较其分数；这些测试集此前已报告，不是全新外部盲测。','',
        table.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        '提升区间来自按测试原文分块的2000次配对bootstrap（seed=20261020），而非把相关token当作独立样本。',
        'original_best_accuracy是原始单层探针在旧prompt协议中的最佳已报告分数，仅作历史参照。',
        'baseline是同一协议的norm-20、C=.005、标准化探针，使用相同训练数据重新拟合。','',
        'selected-probes.pkl含4个新选中探针；原始72个探针与旧tomato图没有覆盖。',
        '新探针的feature_spec规定输入抓取点和拼接顺序，不能直接代入原来的单层norm特征。',
        '原utils.probes.run_projections支持按feature_spec构建的二维特征；数值及消费者检查见consumer-validation.json。',
        'metrics.csv记录所有筛选、完整训练复核及最终拟合指标；screen/refine/final目录保存逐角色召回、混淆矩阵和优化器警告。',
        'selection.json、shortlist.json、splits.json和config.json保存选择过程、精确划分和来源哈希。','',
        '复现：',
        'HF_HOME=/root/.cache/huggingface-role-probe CUDA_VISIBLE_DEVICES=0,1,2,3 ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b .venv/bin/python experiments/role-analysis/extract_feature_variants.py',
        'HF_HOME=/root/.cache/huggingface-role-probe .venv/bin/python experiments/role-analysis/search_feature_variants.py',
        'CUDA_VISIBLE_DEVICES=3 .venv/bin/python experiments/role-analysis/summarize_feature_search.py',
    ]
    (run.OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    complete=json.loads((run.OUT/'complete.json').read_text())
    complete.update(consumer_validation_passed=True,comparison=records,
                    source_sha256={name:run.engine.digest(run.engine.HERE/name) for name in ['extract_feature_variants.py','feature_pipeline_runtime.py','search_feature_variants.py','summarize_feature_search.py']})
    run.atomic_json(run.OUT/'complete.json',complete)
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':main()
