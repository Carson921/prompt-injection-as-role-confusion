"""Assemble validated feature-probe results, fresh-document evidence and bundle."""
import json
from pathlib import Path
import tarfile

import search_feature_variants as run


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out=run.OUT
    main_results=pd.read_csv(out/'comparison.csv')
    fresh=pd.read_csv(out/'fresh/metrics.csv')
    assert json.loads((out/'fresh/complete.json').read_text())['selection_unchanged']
    assert json.loads((out/'consumer-validation.json').read_text())['all_passed']
    rows=[]
    for r in main_results.to_dict('records'):
        rows.append(dict(dataset=r['protocol'],role_space=r['role_space'],
                         baseline_accuracy=r['baseline_accuracy'],selected_accuracy=r['selected_accuracy'],
                         gain_percentage_points=r['gain_percentage_points'],
                         gain_ci95_lower=r['gain_ci95_lower'],gain_ci95_upper=r['gain_ci95_upper'],
                         test_documents=r['test_documents']))
    for name in run.engine.ROLE_SPACES:
        group=fresh[fresh.role_space==name]
        before=group[group.variant=='baseline'].iloc[0]
        after=group[group.variant=='selected'].iloc[0]
        ci=json.loads((out/f'fresh/{name}-comparison.json').read_text())
        rows.append(dict(dataset='fresh-c4',role_space=name,baseline_accuracy=float(before.accuracy),
                         selected_accuracy=float(after.accuracy),**{k:ci[k] for k in
                         ['gain_percentage_points','gain_ci95_lower','gain_ci95_upper','test_documents']}))
    table=pd.DataFrame(rows)
    table.to_csv(out/'all-test-comparisons.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5))
    y=np.arange(len(table))
    ax.errorbar(table.gain_percentage_points,y,xerr=np.vstack([
        table.gain_percentage_points-table.gain_ci95_lower,
        table.gain_ci95_upper-table.gain_percentage_points]),fmt='o',capsize=4,color='#0084d1')
    ax.axvline(0,color='grey',linestyle='--',linewidth=1)
    labels=[f'{r.dataset} / {"4" if r.role_space=="suca" else "5"} roles' for r in table.itertuples()]
    ax.set(yticks=y,yticklabels=labels,xlabel='Accuracy gain over norm-20 baseline (percentage points)',
           title='Layers 16 + 20 + 24 pre-normalization residual features')
    ax.invert_yaxis()
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'gain-confidence.{suffix}',dpi=180)
    plt.close(fig)
    pngs=sorted((out/'tomato/plots').rglob('*.png'))
    pdfs=sorted((out/'tomato/plots').rglob('*.pdf'))
    assert len(pngs)==len(pdfs)==8
    for p in pngs:
        with Image.open(p) as im:im.verify()
    for p in pdfs:
        with p.open('rb') as f:assert f.read(5)==b'%PDF-'
    assert json.loads((out/'tomato/extraction.json').read_text())['exact_original_norm20']
    assert json.loads((out/'tomato/projection-validation.json').read_text())['probability_sums_checked']
    projections=pd.read_csv(out/'tomato/projections.csv.gz',keep_default_na=False)
    keys=['model_variant','role_space','prompt_key','sample_ix']
    chosen=projections.loc[projections.groupby(keys).prob.idxmax()].copy()
    chosen['correct']=chosen.target_role==chosen.base_message_type
    role_accuracy=chosen.groupby(['model_variant','role_space','prompt_key','base_message_type'],as_index=False).agg(
        tokens=('correct','size'),accuracy=('correct','mean'))
    role_accuracy.to_csv(out/'tomato/role-accuracy.csv',index=False)
    chosen.groupby(['model_variant','role_space','prompt_key'],as_index=False).agg(
        tokens=('correct','size'),accuracy=('correct','mean')).to_csv(out/'tomato/overall-accuracy.csv',index=False)
    cot=role_accuracy[(role_accuracy.role_space=='suca')&(role_accuracy.base_message_type=='cot')]
    assert run.engine.digest(run.engine.HERE/'outputs/probes/gptoss-120b.pkl')==json.loads((out/'config.json').read_text())['source_probes_sha256']
    frozen=json.loads((out/'fresh/frozen-selection.json').read_text())
    assert frozen['selection_sha256']==run.engine.digest(out/'selection.json')
    extra=[
        '', '新增100篇C4文档的确认评估',
        '固定来源：allenai/c4，revision 1588ec454efa1a09f29cd18ddd04fe05fc8653a2，English validation shard 0。',
        '40MB原始分片通过公布的SHA256校验。按顺序选取前100个非空且不与原249篇输入正文SHA256重复的文档。',
        '比较的document协议模型在本次新文本评估前已冻结；没有用新文本结果选择特征、层或C。',
        '每篇文档仍配对渲染五种角色，system正常保留。100篇文本只作评估，没有加入本轮训练。',
        '这是一项C4来源的新增文本确认，不能直接当作多轮对话准确率或跨所有来源的准确率。',
        '本地原文、输入和token缓存位于fresh/cache/，不包含在发布结果中。公开的document-keys.json记录行号、URL及正文哈希。',
        '',table.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        '新版tomato对照',
        'tomato/plots/baseline和tomato/plots/selected各保存三种格式的四面板图及三格式CoT图（PNG/PDF）。',
        '两边均使用document划分训练的探针，文本完全沿用原20B固定示例，未用tomato调整参数。',
        '新抓取的tomato norm-20逐值匹配旧缓存；概率及图片文件检查通过。',
        '实测限制：单消息测评收益没有转化为此多轮示例的CoT识别改善。',
        '例如四角色proper_tags的179个CoT内容token，新探针召回率为0%，基线约1.68%。',
        '多轮上下文泛化需要单独训练/验证；没有围绕tomato继续选择参数或调整概率。',
        cot.to_string(index=False,float_format=lambda x:f'{x:.6f}'),
        '', '运行与复现记录',
        '完整宽特征矩阵切换曾触发显存分配失败；改为每个特征方案独立子进程，以释放库保留的分配。',
        '数学处理、划分及已完成的候选记录保持一致；config.json同时保存修改前后源代码哈希。',
        '按均匀分块计算StandardScaler统计量，预检查相对方差差异低于2e-7，见loader-validation.json。',
        '结果包：gptoss-120b-feature-probes.tar.gz；对应.sha256提供下载校验。',
        '新抓取激活和临时日志保留为本地缓存，没有放入结果包。原始72个探针和旧tomato图保留。',
        '', '新增文档评估复现（先完成原特征实验）：',
        'HF_HOME=/root/.cache/huggingface-role-probe CUDA_VISIBLE_DEVICES=0,1,2,3 ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b .venv/bin/python experiments/role-analysis/evaluate_fresh_feature_probes.py',
        'HF_HOME=/root/.cache/huggingface-role-probe CUDA_VISIBLE_DEVICES=0,1,2,3 ROLE_ANALYSIS_LOAD_PATH=/dev/shm/role-analysis-gptoss120b .venv/bin/python experiments/role-analysis/tomato_feature_probes.py',
        '.venv/bin/python experiments/role-analysis/finalize_feature_search.py',
    ]
    report=out/'report.txt'
    text=report.read_text().split('\n新增100篇C4文档的确认评估')[0].rstrip()
    report.write_text(text+'\n'+'\n'.join(extra)+'\n')
    scripts=['extract_feature_variants.py','feature_pipeline_runtime.py','search_feature_variants.py',
             'summarize_feature_search.py','evaluate_fresh_feature_probes.py','tomato_feature_probes.py',
             'plot_feature_tomato.R','finalize_feature_search.py']
    complete=json.loads((out/'complete.json').read_text())
    complete.update(fresh_document_validation_passed=True,fresh_documents=100,
                    fresh_comparisons=[r for r in rows if r['dataset']=='fresh-c4'],
                    tomato_png=8,tomato_pdf=8,
                    source_sha256={name:run.engine.digest(run.engine.HERE/name) for name in scripts})
    run.atomic_json(out/'complete.json',complete)
    run.atomic_json(out/'fresh/status.json',dict(stage='complete',documents=100,selection_unchanged=True))
    run.atomic_json(out/'tomato/complete.json',dict(png=8,pdf=8,probability_sums_checked=True,original_norm20_exact=True))
    run.atomic_json(out/'tomato/status.json',dict(stage='complete',png=8,pdf=8))
    archive=out/'gptoss-120b-feature-probes.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in sorted(out.rglob('*')):
            rel=path.relative_to(out)
            if not path.is_file() or rel.parts[0] in ['activations','logs'] or str(rel).startswith('fresh/cache/'):
                continue
            if path==archive or path.name.endswith('.sha256') or path.suffix in ['.pt','.log']:
                continue
            tar.add(path,arcname=str(Path('results')/rel),recursive=False)
        for name in scripts+['tune_gptoss120b_probes.py','run_local_gptoss120b.py']:
            tar.add(run.engine.HERE/name,arcname=f'source/{name}')
    checksum=run.engine.digest(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(checksum+'  '+archive.name+'\n')
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames()
        assert not any('/activations/' in n or '/cache/' in n or n.endswith('.pt') or n.endswith('.log') for n in names)
        assert tar.extractfile('results/complete.json').read()==(out/'complete.json').read_bytes()
        assert tar.extractfile('results/selected-probes.pkl').read()==(out/'selected-probes.pkl').read_bytes()
    print(table.to_string(index=False),flush=True)
    print('Bundle',archive,'SHA256',checksum,flush=True)


if __name__=='__main__':main()
