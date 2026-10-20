"""Build an auditable report, aggregate figures, gallery index, and result bundle."""
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

import systemness_data as data


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    output=data.OUT
    selected=json.loads((output/'selection.json').read_text())
    assert selected['test_used_for_selection'] is False
    assert all(data.digest(output/'models'/name)==sha for name,sha in selected['model_hashes'].items())
    frames=[]; comparisons=[]; docs=[]; positions=[]
    for part in ['test','challenge']:
        folder=output/'evaluation'/part
        complete=json.loads((folder/'complete.json').read_text())
        assert complete['selection_sha256']==data.digest(output/'selection.json')
        assert complete['test_weights_unchanged'] and complete['formal_and_semantic_labels_separate']
        frames.append(pd.read_csv(folder/'metrics.csv'))
        comparisons.append(pd.read_csv(folder/'paired-comparisons.csv'))
        docs.append(pd.read_csv(folder/'document-counts.csv'))
        positions.append(pd.read_csv(folder/'system-by-position.csv'))
    metrics=pd.concat(frames,ignore_index=True)
    metrics.to_csv(output/'test-metrics.csv',index=False)
    changes=pd.concat(comparisons,ignore_index=True)
    changes.to_csv(output/'paired-test-comparisons.csv',index=False)
    position=pd.concat(positions,ignore_index=True)
    position.to_csv(output/'system-by-position.csv',index=False)
    chosen=metrics[(metrics.model=='selected')&(metrics.wrapper=='native')].copy()
    controls=metrics[(metrics.model=='previous-selected')&(metrics.wrapper=='native')].copy()
    profiles=['single','multi','natural','long-single','natural-unseen-system']
    labels=['C4 single','C4 multi','Natural dialogue','C4 <=1024 tokens','New system wording']
    chosen=chosen.set_index('profile').loc[profiles].reset_index()
    controls=controls.set_index('profile').loc[profiles].reset_index()
    plots=output/'plots'
    plots.mkdir(exist_ok=True)
    x=np.arange(len(profiles)); width=.36
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for ax,metric,title in zip(axes,['system_precision','system_recall','system_balanced_accuracy'],
                              ['System precision','System recall','Binary balanced accuracy']):
        ax.bar(x-width/2,controls[metric]*100,width,label='Previous three-layer probe',color='#90a1b9')
        ax.bar(x+width/2,chosen[metric]*100,width,label='New validation-selected probe',color='#0084d1')
        ax.axhline(90,color='grey',linestyle='--',linewidth=.8)
        ax.set(ylim=(0,103),ylabel=title+' (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,ncol=2)
    axes[-1].set_xticks(x,labels,rotation=15,ha='right')
    fig.suptitle('GPT-OSS-120B MXFP4 | systemness on held-out native inputs')
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(plots/f'systemness-comparison.{suffix}',dpi=180)
    plt.close(fig)

    gains=changes[(changes.wrapper=='native')&(changes.metric=='system_f1')].set_index('profile').loc[profiles]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.errorbar(gains.gain_percentage_points,np.arange(len(gains)),
        xerr=np.vstack([gains.gain_percentage_points-gains.ci95_low,gains.ci95_high-gains.gain_percentage_points]),fmt='o',capsize=4)
    ax.axvline(0,color='grey',linestyle='--',linewidth=.8)
    ax.set(yticks=np.arange(len(gains)),yticklabels=labels,xlabel='System F1 gain (percentage points)',
           title='Paired 95% bootstrap intervals by document or system wording')
    ax.invert_yaxis();fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(plots/f'system-f1-gain.{suffix}',dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4))
    for model,color in [('previous-selected','#90a1b9'),('selected','#0084d1')]:
        frame=position[(position.model==model)&(position.profile=='long-single')].sort_values('start')
        ax.plot(np.arange(len(frame)),frame.system_recall*100,marker='o',color=color,label=model)
        ticks=[f'{int(r.start)}-{int(r.end)}' for r in frame.itertuples()]
    ax.set(xticks=np.arange(len(ticks)),xticklabels=ticks,ylim=(0,103),xlabel='Content token position',
           ylabel='System recall (%)',title='Systemness retention across long system messages')
    ax.legend(frameon=False);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(plots/f'long-system-recall.{suffix}',dpi=180)
    plt.close(fig)

    cases=['fixed-small-single','fixed-large-single','fixed-small-mixed','fixed-large-mixed']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['system_f1','accuracy'],['System F1','Five-role accuracy']):
        for i,profile in enumerate(['single','multi','natural','long-single']):
            frame=metrics[(metrics.model.isin(cases))&(metrics.profile==profile)&(metrics.wrapper=='native')].set_index('model').loc[cases]
            ax.plot(np.arange(4),frame[metric]*100,marker='o',label=profile)
        ax.set(xticks=np.arange(4),xticklabels=['249 / single','1200 / single','249 / mixed','1200 / mixed'],ylabel=title+' (%)')
        ax.tick_params(axis='x',rotation=20)
    axes[1].legend(frameon=False,fontsize=8);fig.tight_layout()
    fig.suptitle('Fixed linear readout, C=.005; scalers fit on each training pool',fontsize=10)
    fig.tight_layout(rect=[0,0,1,.95])
    for suffix in ['png','pdf']:fig.savefig(plots/f'data-context-ablation.{suffix}',dpi=180)
    plt.close(fig)

    gallery=json.loads((output/'gallery.json').read_text())
    assert gallery['png']==gallery['pdf']==36
    pngs=list((plots/'gallery').glob('*.png'));pdfs=list((plots/'gallery').glob('*.pdf'))
    assert len(pngs)==len(pdfs)==36
    for path in plots.rglob('*.png'):
        with Image.open(path) as im:im.verify()
    for path in plots.rglob('*.pdf'):
        with path.open('rb') as f:assert f.read(5)==b'%PDF-'
    index=['# Batch role projection gallery','',
           '12 conversation IDs chosen at equal spacing before inspecting example accuracy. All 144 held-out conversations contribute to the metrics.',
           '', 'Colors represent the original message function. Formal role labels follow the actual wrapper. Untagged text has no claimed formal role accuracy.', '']
    for group in gallery['conversations']:
        index.append(f'- {group}: '+ ' | '.join(f'[{wrapper}](gallery/{group}-{wrapper}.png)' for wrapper in gallery['formats']))
    (plots/'gallery-index.md').write_text('\n'.join(index)+'\n')

    summary_fields=['profile','groups','tokens','system_precision','system_recall','system_f1','system_balanced_accuracy',
                    'system_false_positive_rate','accuracy','role_recall_system','role_recall_cot']
    summary=chosen[summary_fields]
    summary.to_csv(output/'selected-summary.csv',index=False)
    negatives=metrics[(metrics.model.isin(['previous-selected','selected']))&(metrics.wrapper=='all-user')]
    checks=[]
    for row in chosen.to_dict('records'):
        checks.append(dict(profile=row['profile'],system_precision_ge_90=row['system_precision']>=.90,
                           system_recall_ge_90=row['system_recall']>=.90,
                           system_balanced_accuracy_ge_90=row['system_balanced_accuracy']>=.90))
    scripts=['systemness_data.py','systemness_challenges.py','systemness_extract.py','systemness_probes.py',
             'systemness_boosting.py','systemness_lexical.py','systemness_radial.py','systemness_evaluate.py','validate_systemness_data.py',
             'finalize_systemness.py','feature_pipeline_runtime.py',
             'run_local_gptoss120b.py','requirements-systemness.txt','requirements-local-4xa40.txt']
    candidate_count=len(list((output/'candidates').glob('*.json')))
    complete=dict(candidates=candidate_count,c4_train_documents=1200,c4_validation_documents=200,c4_test_documents=400,
                  natural_validation_conversations=40,natural_test_conversations=144,
                  long_test_documents=64,long_documents_subset_of_c4_test=True,
                  natural_matched_system_wording_test=True,unseen_system_wordings=32,
                  test_selection_frozen=True,original_probes_unchanged=(data.digest(data.HERE/'outputs/probes/gptoss-120b.pkl')==
                      json.loads((output/'data-complete.json').read_text())['original_probes_sha256']),
                  all_finite=True,plots_validated=True,gallery_png=36,gallery_pdf=36,working_target_checks=checks,
                  source_sha256={name:data.digest(data.HERE/name) for name in scripts},
                  selected_candidate=selected['candidate'],selection_sha256=data.digest(output/'selection.json'))
    data.atomic_json(output/'complete.json',complete)
    report=[
        'GPT-OSS-120B MXFP4：扩大数据、上下文覆盖与systemness探针实验','',
        '语言模型权重保持冻结，特征提取使用四张A40。输入为第16/20/24层归一化前残差的拼接。',
        'system作为正常训练类别保留，训练/指标均不计入header和结构tag token。',
        '新C4原文1800篇：1200训练、200验证、400测试；排除原249篇和上轮100篇正文的精确SHA256重复。',
        '四组对照：249或1200篇 × 单消息或单消息加多轮。另在相同文档划分内加入最长1024 token的正文。',
        '数据/上下文消融图固定使用同一线性读出设置C=.005、system权重1；同时也报告每组验证集选中的最佳读出。',
        '训练每篇/每角色/每种profile最多32个均匀抽取内容token；验证最多64个；native测试保留全部内容token。',
        '无tag或user包装中，跨越原消息边界的少量BPE token语义归属不明确，因此不纳入投影或计分。',
        '自然对话来自仓库现有GPT-OSS-20B生成数据，40段验证、144段测试；这些不是新生成的120B回答。',
        '长文本64篇是400篇C4测试集的子集；新system措辞是同144段自然对话的配对变体，不能重复计为新增独立文档/对话。',
        '自然对话的消息正文每条截取前192 token，因此结果不代表任意长度的完整会话。',
        '32种新system措辞只进入测试；原自然对话使用8种措辞，因此system召回的不确定性按措辞分组估计。',
        'C4是同一固定验证分片的新增原文，不是所有领域的跨来源泛化结论；仅精确正文、完整正文和URL重复已检查。','',
        '选择过程',
        f'四组数据设置先各比较4个线性配置、PCA-512/1024、MLP-64/128和分层MLP-128，共36个；再在large-mixed追加GPU树分类器、词汇控制与幅度/方向分离读出，全轮共{candidate_count}个候选。',
        '词汇控制读出比较移除16/64/128个相关方向的线性模型，以及移除64个方向的MLP。',
        '词汇方向仅使用训练集单消息/长正文的五角色同词配对样本与冻结token embedding估计，固定投影折叠进读出的第一层权重。推断不额外使用token ID、tag或真实角色。',
        '幅度/方向读出仅使用当前token的三层残差：结合原始值、各层RMS归一化方向和三个log RMS幅度；所有缩放统计量只拟合训练部分。',
        '分类阈值和候选只使用四种内部验证profile；每个profile的system误报率约束<=5%，最大化平均system F1。',
        '同时要求平均五角色验证accuracy不低于上轮三层探针超过1个百分点。基线也用相同验证规则校准阈值。',
        '分层分类器先建模system/非system，再建模其余四类；MLP属于非线性读出，不能把其收益全部归因于线性可分性。',
        '树分类器使用原始缓存特征，不做缩放；比较五分类树，以及独立system二分类树与固定四类条件线性读出的组合。',
        f'选中：{selected["candidate"]}；system阈值：{selected["system_threshold"]:.8f}。',
        'system_precision/recall/F1是固定阈值的二分类指标；role_recall_system与accuracy使用五分类argmax，二者不能混称。','',
        '选中模型的独立测试结果（全部native格式）',summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'),'',
        '相对基线的system F1变化（95%配对分块bootstrap）',
        gains.reset_index().to_string(index=False,float_format=lambda v:f'{v:.6f}'),'',
        '全部放入user标签时的system误报：真实结构角色为user，原始system-like文字只作语义标记。',
        negatives[['model','profile','groups','tokens','system_false_positive_rate']].to_string(index=False,float_format=lambda v:f'{v:.6f}'),'',
        '未加tag的文本只保存消息功能投影，不报告结构角色准确率。没有把原始system样式标签当成user包装下的真实结构角色。',
        '置信区间使用2000次配对bootstrap：C4按原文，自然对话按system提示模板聚类；不是token独立抽样。',
        'test-metrics.csv记录全部模型和格式；paired-test-comparisons.csv记录分组区间；system-by-position.csv显示长system信号衰减。',
        'plots/gallery-index.md提供12段不同对话、三种格式共36张四面板图，全部144段测试对话都纳入数值评估。','',
        '阶段目标检查（precision、recall及balanced accuracy分别>=90%，不等同于用户最终验收）',
        json.dumps(checks,ensure_ascii=False,indent=2),'',
        '复现顺序（HF_HOME=/root/.cache/huggingface-role-probe，ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b）：',
        '1. systemness_data.py；systemness_challenges.py',
        '2. systemness_extract.py --part development；--part augmentation（CUDA_VISIBLE_DEVICES=0,1,2,3）',
        '3. systemness_probes.py --stage search；systemness_boosting.py；systemness_lexical.py --prepare-embedding；systemness_lexical.py；systemness_radial.py；systemness_probes.py --stage select',
        '4. systemness_extract.py --part test；--part challenge（四卡）',
        '5. systemness_evaluate.py --part test；--part challenge；--gallery（单卡）',
        '6. finalize_systemness.py','',
        '所有脚本位于experiments/role-analysis/，使用.venv/bin/python运行。结果包不含基础模型权重、激活、原文和临时训练缓存。',
        '依赖见requirements-systemness.txt。安装时为UV_CACHE_DIR和PIP_CACHE_DIR指定可访问的本地缓存目录。',
    ]
    (output/'report.txt').write_text('\n'.join(report)+'\n')
    archive=output/'gptoss-120b-systemness-results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in sorted(output.rglob('*')):
            relative=path.relative_to(output)
            if not path.is_file() or relative.parts[0] in ['cache','logs','checkpoints']:
                continue
            if path==archive or path.name.endswith('.sha256'):
                continue
            tar.add(path,arcname=str(Path('results')/relative),recursive=False)
        for name in scripts:
            tar.add(data.HERE/name,arcname=f'source/{name}')
    sha=data.digest(archive)
    archive.with_suffix('.gz.sha256').write_text(sha+'  '+archive.name+'\n')
    print(summary.to_string(index=False),flush=True)
    print('Bundle SHA256:',sha,flush=True)


if __name__=='__main__':main()
