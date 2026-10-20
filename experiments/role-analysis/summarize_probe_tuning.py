"""Validate and summarize the separate cached-activation tuning experiment."""
import hashlib
import json
from pathlib import Path
import pickle
import sys

import tune_gptoss120b_probes as tune


def validate_consumers():
    import cupy as cp
    import numpy as np
    import pandas as pd
    import torch
    from utils.probes import run_projections
    labels = pd.read_parquet(tune.SOURCE / 'training_labels.parquet')
    inputs = pd.read_parquet(tune.SOURCE / 'training_inputs.parquet').set_index('prompt_ix')
    count = json.loads((tune.SOURCE / 'manifest.json').read_text())['tokens']
    models = []
    with (tune.OUT / 'selected-probes.pkl').open('rb') as f:
        probes = pickle.load(f)
    for probe in probes:
        record = probe['tuning']
        assert record['actual_classifier_dtype'] == 'float32'
        train_ids, test_ids = set(probe['train_prompt_ids']), set(probe['test_prompt_ids'])
        assert not train_ids & test_ids
        train_docs = set(inputs.loc[list(train_ids)].question_ix)
        test_docs = set(inputs.loc[list(test_ids)].question_ix)
        if record['protocol'] == 'document':
            assert not train_docs & test_docs
        test = labels[labels.prompt_ix.isin(test_ids)]
        sample = test.groupby('role', group_keys=False).head(32).sort_values('sample_ix')
        mmap = torch.from_file(str(tune.SOURCE / f'activations/layer-{probe["layer_ix"]}.f16'),
                               shared=False, size=count*2880, dtype=torch.float16).view(-1,2880)
        features = mmap[sample.sample_ix.tolist()].clone()
        sample = sample.assign(sample_ix=range(len(sample)))
        got = run_projections(sample, features, probe)
        direct = cp.asnumpy(probe['probe'].predict_proba(cp.asarray(features.float().numpy())))
        expected = got.pivot(index='sample_ix', columns='target_role', values='prob')[probe['role_space']].to_numpy()
        assert np.isfinite(expected).all()
        np.testing.assert_allclose(expected.sum(axis=1), 1, atol=2e-6)
        difference = float(np.abs(direct-expected).max())
        assert difference < 1e-4, difference
        models.append(dict(protocol=record['protocol'],role_space=record['role_space'],
                           layer=probe['layer_ix'],rows=len(sample),
                           original_consumer_interface=True,max_probability_difference=difference,
                           train_test_prompt_disjoint=True,train_test_document_disjoint=not bool(train_docs&test_docs)))
    tune.atomic_json(tune.OUT / 'consumer_validation.json',dict(models=models,all_passed=True))
    dtypes=[]
    for path in sorted((tune.OUT/'models').glob('*.pkl')):
        with path.open('rb') as f:
            saved=pickle.load(f)
        clf=saved['probe'].named_steps['clf']
        observed=[str(clf.dtype),str(clf.coef_.dtype),str(clf.solver_model.dtype)]
        assert observed==[saved['tuning']['dtype']]*3
        dtypes.append(dict(model=path.name,classifier_dtype=observed[0],
                           coefficient_dtype=observed[1],solver_dtype=observed[2]))
    assert len(dtypes)==10
    tune.atomic_json(tune.OUT/'dtype_validation.json',dict(models=dtypes,all_passed=True))


def summarize():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final = pd.read_csv(tune.OUT / 'final_metrics.csv')
    original = pd.read_csv(tune.SOURCE / 'training_metrics.csv')
    comparisons = []
    for protocol in tune.PROTOCOLS:
        for name in tune.ROLE_SPACES:
            group = final[(final.protocol==protocol)&(final.role_space==name)]
            selected = group[(group.phase=='selected')&(group.dtype=='float32')].iloc[0]
            baseline = group[group.phase=='baseline'].iloc[0]
            original_best = original[original.role_space==name].acc.max()
            result = dict(protocol=protocol,role_space=name,layer=int(selected.layer),
                          selected_scaling=bool(selected.scaling),selected_C=float(selected.C),
                          baseline_accuracy=float(baseline.test_accuracy),selected_accuracy=float(selected.test_accuracy),
                          gain_percentage_points=float(100*(selected.test_accuracy-baseline.test_accuracy)),
                          baseline_cot_recall=float(baseline.recall_cot),selected_cot_recall=float(selected.recall_cot),
                          baseline_system_recall=float(baseline.recall_system),selected_system_recall=float(selected.recall_system),
                          original_protocol_best_accuracy=float(original_best))
            if protocol=='document':
                precision = group[group.dtype=='float64'].iloc[0]
                result.update(fp64_accuracy=float(precision.test_accuracy),
                              fp64_gain_percentage_points=float(100*(precision.test_accuracy-selected.test_accuracy)))
            comparisons.append(result)
    table=pd.DataFrame(comparisons)
    table.to_csv(tune.OUT / 'comparison.csv',index=False)
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax, protocol in zip(axes,tune.PROTOCOLS):
        rows=table[table.protocol==protocol]
        x=list(range(len(rows)))
        before=ax.bar([v-.18 for v in x],rows.baseline_accuracy,width=.36,label='C=.005, scaling on')
        after=ax.bar([v+.18 for v in x],rows.selected_accuracy,width=.36,label='Validation-selected')
        ax.bar_label(before,labels=[f'{v:.2%}' for v in rows.baseline_accuracy],fontsize=8,padding=3)
        ax.bar_label(after,labels=[f'{v:.2%}' for v in rows.selected_accuracy],fontsize=8,padding=3)
        ax.set(xticks=x,xticklabels=rows.role_space,title=protocol,ylabel='Held-out token accuracy',ylim=(0,1))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(tune.OUT/'test-comparison.png',dpi=180)
    fig.savefig(tune.OUT/'test-comparison.pdf')
    plt.close(fig)
    lines=[
        'GPT-OSS-120B cached-activation probe tuning / 角色探针调参结果', '',
        '模型权重继续为冻结的 MXFP4。本轮没有重新运行或修改语言模型，原始72个探针及tomato图保持不变。',
        '搜索四角色SUCA和五角色SUCAT、第16/20层、标准化开/关、C={.0005,.005,.05,.5,5}。',
        '两个协议各40个候选，总计80个；按内部验证accuracy、NLL选择，不按测试指标选配置。',
        'legacy：保留原始prompt级测试划分；原训练集内部按原文分组留15%作验证。',
        'document：按249篇原文分成199/25/25篇训练/验证/测试；选参后用224篇重新拟合。',
        '原文分组后，同一篇原文的五个角色版本始终处于同一分区，system样本正常保留。',
        '两种协议的分数不能直接相减；原协议的测试结果此前已报告，并非全新外部盲测。', '',
        table.to_string(index=False,float_format=lambda x:f'{x:.6f}'), '',
        '结论：本轮没有得到稳定的accuracy提升，继续保留原探针；原五角色最佳验证accuracy为68.8241%。',
        '五角色选中的超参数与默认参数相同，微小分数差异来自相同配置的重复拟合，不属于调参收益。',
        'document协议FP64相对FP32的四角色变化约-0.0110个百分点、五角色约+0.0050个百分点。',
        '本次FP64拟合耗时约为对应FP32的11至13倍。结果不支持优先投入更多FP64拟合。',
        '这一结论限于已测网格和单次内部验证划分，未证明其他数据或特征方案没有提升空间。', '',
        'baseline在各自选中层上，以C=.005、标准化开启重新拟合；用于隔离参数改动。',
        'original_protocol_best_accuracy是旧实验最佳层的已报告分数，仅与legacy协议对应。',
        'FP64仅对document协议已选中的两个配置做精度对照，参数及数据划分均不改变。',
        '实际classifier、coef和solver dtype均有断言；模型权重精度不受探针FP64影响。',
        '拟合的训练accuracy和测试accuracy均记录；逐角色召回、macro-F1、NLL、混淆矩阵和优化器警告见final/*.json。',
        '消费者接口以每个角色32个测试token检查，见consumer_validation.json。', '',
        'selected-probes.pkl：四个按验证集选择的FP32探针（两种协议各两个）。',
        'models/：对应baseline、selected和FP64诊断探针，兼容原utils.probes.run_projections接口。',
        'search_metrics.csv、validation-search.png：全部候选的内部验证结果。',
        'final_metrics.csv、comparison.csv、test-comparison.png：重新拟合后的测试结果。',
        'splits.json、config.json：精确划分、随机种子、超参数及来源哈希。', '',
        '复现：HF_HOME=/root/.cache/huggingface-role-probe .venv/bin/python experiments/role-analysis/tune_gptoss120b_probes.py',
        '验证及报告：CUDA_VISIBLE_DEVICES=3 .venv/bin/python experiments/role-analysis/summarize_probe_tuning.py',
        '所需缓存：原训练activations/layer-16.f16、layer-20.f16及training_{inputs,labels}.parquet。',
        '环境沿用requirements-local-4xa40.txt；激活和临时拟合日志不提交Git。',
    ]
    (tune.OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    complete=json.loads((tune.OUT/'complete.json').read_text())
    complete.update(consumer_validation_passed=True,comparison=comparisons,
                    summary_source_sha256=tune.digest(Path(__file__)))
    tune.atomic_json(tune.OUT/'complete.json',complete)
    print(table.to_string(index=False))


if __name__=='__main__':
    validate_consumers()
    summarize()
