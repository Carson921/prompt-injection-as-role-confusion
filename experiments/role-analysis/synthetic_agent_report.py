"""Generate a factual report from committed six-role experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROLES = ['system', 'developer', 'user', 'cot', 'assistant', 'tool']


def number(value):
    return 'N/A' if value is None else f'{100 * value:.2f}%'


def interval(evaluation, metric):
    value = evaluation.get('confidence_intervals', {}).get('intervals', {}).get(metric)
    return '' if value is None else f"（95% CI {number(value['lower'])}–{number(value['upper'])}）"


def report(run_dir, data_dir, phase='pilot'):
    root = Path(run_dir)
    result = {'phase': phase, 'completed': [], 'missing': []}
    artifacts = {}
    names = ['config.json', f'data-{phase}-manifest.json', 'preflight.json', 'training.json',
             'head-selection.json', 'selection.json', 'freeze.json',
             'resume-validation.json', 'evaluations/dev.json']
    if phase == 'formal':
        names += ['evaluations/test.json', 'evaluations/external.json']
    for name in names:
        path = root / name
        if path.exists():
            artifacts[name] = json.loads(path.read_text())
            result['completed'].append(name)
        else:
            result['missing'].append(name)
    lines = [f'# GPT-OSS-120B 六角色实验：{phase}', '',
             '此报告只陈述本运行目录已有产物；小试 DEV 结果不属于正式 TEST 结果。', '']
    preflight = artifacts.get('preflight.json')
    if preflight:
        lines += [f"GPU 预检状态：{preflight.get('status')}。最大输入 {preflight.get('maximum_input_tokens')} token。",
                  f"全设备峰值 MiB：{preflight.get('device_wide_peak_mib')}。", '']
    data = artifacts.get(f'data-{phase}-manifest.json')
    if data:
        lines += ['数据统计：', '', '| 划分 | 来源文章 | 输入快照 | 正文特征行 | 输入 token |', '|---|---:|---:|---:|---:|']
        for split, stats in data['statistics'].items():
            lines.append(f"| {split} | {stats['sources']} | {stats['snapshots']} | {stats['rows']} | {stats['input_tokens']} |")
        audit_path = Path(data_dir) / phase / 'article-audit.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        lines += ['', f"文章审计状态：{audit.get('status', data.get('article_audit_status'))}；结论仅覆盖所记录的固定分层样本。", '', data.get('filter_limitation', ''), '']
        lines += ['严格过滤排除问号、直接引语、第一/二人称表达和疑似评论/访谈/论坛来源，因而保留下来的文本偏向第三人称说明文章。Dolma3 使用固定版本的六个文章主题组轮转，并非整个 Dolma3 分布的随机样本。', '',
                  '| 划分 | 单消息缓存份额 | 单轮缓存份额 | 多轮缓存份额 | 排除的跨边界 token |', '|---|---:|---:|---:|---:|']
        for split, stats in data['statistics'].items():
            shares = stats.get('saved_profile_proportions', {})
            lines.append(f"| {split} | {number(shares.get('single'))} | {number(shares.get('one_round'))} | {number(shares.get('multi_round'))} | {stats.get('excluded_boundary_tokens', 'N/A')} |")
        lines += ['', '原始正文特征行与家族归一化后的有效权重分别保存在数据清单中；重复快照不自动增加相同事件正文的训练权重。', '']
    heads = artifacts.get('head-selection.json')
    if heads:
        lines += [f"16 头固定顺序：`{', '.join(map(str, heads['heads']))}`。选头仅使用 pilot FIT 内部来源划分；DEV、TEST 和旧分类器不参与。", '']
    selection = artifacts.get('selection.json')
    if selection:
        lines += [f"DEV 选择：`{selection['candidate_id']}`；各类 precision/recall 90% 门槛：{selection['dev_floor_met']}。", '']
        lines += ['模型与最佳 epoch 只按 DEV 选择：先优先考虑六类汇总 precision、recall 均≥90% 的候选，再依次比较更高的结构等权 macro-F1、更高的最弱 precision/recall、更低的 NLL、固定候选 ID。表中的总体 macro-F1 不是第一排序量；不使用合成 TEST 或外部对话挑选模型。', '']
        training = artifacts.get('training.json', {})
        options = training.get('options', {})
        if options:
            lines += [f"分类器最多运行 {options.get('max_epochs')} epochs、至少 {options.get('min_epochs')} epochs，早停 patience={options.get('patience')}；batch={options.get('batch_size')}，AdamW 学习率={options.get('learning_rate')}、weight decay={options.get('weight_decay')}，MLP dropout={options.get('dropout')}，梯度裁剪={options.get('gradient_clip')}。标准化仅由 FIT 估计，梯度更新只使用 FIT。", '']
        lines += ['| 候选 | 最佳 epoch | 已运行 epochs | DEV macro-F1 | 结构等权 macro-F1 | 最弱 precision/recall |', '|---|---:|---:|---:|---:|---:|']
        for candidate in selection['candidate_results']:
            dev = candidate['dev']
            lines.append(f"| {candidate['candidate_id']} | {candidate['best_epoch']} | {candidate['epochs']} | {number(dev['overall']['macro_f1'])} | {number(dev['structure_equal_macro_f1'])} | {number(dev['minimum_precision_recall'])} |")
        lines += ['']
        sampling = training.get('sampler_protocol', {})
        weights = sampling.get('profile_weights')
        if sampling.get('mode') == 'profile_then_structure_then_present_role_then_family' and weights:
            target = '/'.join(number(weights.get(profile)) for profile in ('single', 'one_round', 'multi_round'))
            lines += [f'实际梯度采样先按单消息/单轮/多轮 {target} 分层，再在层内均匀选择结构、适用角色，最后按正文家族归一化权重选行。未叠加类别损失权重。', '']
        else:
            lines += ['本运行未记录标准分层采样协议；不能仅凭缓存份额声称梯度抽样达到 20%/30%/50%。具体采样配置见 training.json。', '']
        lines += ['下表累计计数涵盖候选实际执行的全部 epochs；导出权重恢复自该候选的最佳 epoch，较晚执行的更新不会加入较早的最佳 checkpoint。', '',
                  '| 候选 | 全部已运行 epochs 的抽样行数 | 单消息实际份额 | 单轮实际份额 | 多轮实际份额 |', '|---|---:|---:|---:|---:|']
        for candidate in selection['candidate_results']:
            counts = candidate.get('sampled_rows_by_profile', {})
            total = sum(counts.values())
            shares = {profile: counts.get(profile, 0) / total for profile in ('single', 'one_round', 'multi_round')} if total else {}
            lines.append(f"| {candidate['candidate_id']} | {total if counts else 'N/A'} | {number(shares.get('single'))} | {number(shares.get('one_round'))} | {number(shares.get('multi_round'))} |")
        lines += ['']
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        for candidate in selection['candidate_results']:
            cid = candidate['candidate_id']
            history = json.loads((root / 'candidates' / cid / 'history.json').read_text())
            if isinstance(history, dict):
                history = history.get('history', history.get('epochs', []))
            if cid == selection['candidate_id']:
                selected_counts = {profile: sum(item.get('sampled_rows_by_profile', {}).get(profile, 0)
                                                for item in history if item['epoch'] <= candidate['best_epoch'])
                                   for profile in ('single', 'one_round', 'multi_round')}
                selected_total = sum(selected_counts.values())
                if selected_total:
                    selected_shares = '/'.join(number(selected_counts[profile] / selected_total)
                                               for profile in ('single', 'one_round', 'multi_round'))
                    lines += [f"选中 checkpoint 截至最佳 epoch {candidate['best_epoch']} 实际使用 {selected_total} 次正文行抽样，单消息/单轮/多轮份额为 {selected_shares}。", '']
            values = [h.get('dev', {}).get('structure_equal_macro_f1', h.get('structure_equal_macro_f1')) for h in history]
            if values and all(v is not None for v in values):
                ax.plot(range(1, len(values)+1), values, label=cid)
        ax.set(xlabel='Epoch', ylabel='DEV structure-equal macro-F1', ylim=(0, 1))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root / 'training-curves.png', dpi=160)
        fig.savefig(root / 'training-curves.pdf')
        plt.close(fig)
    for name in ['evaluations/dev.json', 'evaluations/test.json', 'evaluations/external.json']:
        evaluation = artifacts.get(name)
        if evaluation:
            overall = evaluation['overall']
            lines += [f"`{name}`：{overall['tokens']} 个文章正文或真实对话正文位置；accuracy={number(overall['accuracy'])}{interval(evaluation, 'accuracy')}，macro-F1={number(overall['macro_f1'])}{interval(evaluation, 'macro_f1')}。", '',
                      '| 类别 | 正例位置 | Precision | Recall | F1 | 误报数 |', '|---|---:|---:|---:|---:|---:|']
            for role in ROLES:
                row = overall['roles'][role]
                lines.append(f"| {role} | {row['support']} | {number(row['precision'])} | {number(row['recall'])} | {number(row['f1'])} | {row['false_positives']} |")
            lines += ['', '无正例类别的 recall/F1 为 N/A；其误报仍计入。分项完整结果见对应 JSON 的 by_structure 和 by_attribute。', '']
            if name.endswith('test.json'):
                floor = all(overall['roles'][r]['support'] > 0 and
                            (overall['roles'][r]['precision'] or 0) >= .9 and
                            (overall['roles'][r]['recall'] or 0) >= .9 for r in ROLES)
                reached = overall['macro_f1'] >= .95 and floor
                lines += [f"正式合成 TEST 点估计目标是否达到：{'是' if reached else '否'}。阈值为 macro-F1≥95%，六类 precision/recall 均≥90%；置信区间单列，不将其下界称为达标承诺。", '']
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            matrix = np.asarray(overall['confusion'])
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(matrix, cmap='Blues')
            ax.set_xticks(range(6), ROLES, rotation=35, ha='right')
            ax.set_yticks(range(6), ROLES)
            ax.set(xlabel='Predicted role', ylabel='True role', title=name)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                            color='white' if matrix[i, j] > matrix.max()/2 else 'black')
            fig.tight_layout()
            stem = Path(name).stem + '-confusion'
            fig.savefig(root / (stem + '.png'), dpi=160)
            fig.savefig(root / (stem + '.pdf'))
            plt.close(fig)
            lines += [f'![{name} 混淆矩阵]({stem}.png)', '']
            for key in ['profile', 'template_id', 'rounds', 'snapshot_stage', 'history_policy', 'position_bin']:
                values = evaluation.get('by_attribute', {}).get(key, {})
                if values:
                    lines += [f'{key} 分项：', '', '| 分组 | 位置数 | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
                    for group, score in values.items():
                        lines.append(f"| {group} | {score['tokens']} | {number(score['accuracy'])} | {number(score['macro_f1'])} |")
                    lines += ['']
            from synthetic_agent_position_diagnostics import evaluate_positions
            evaluation_name = Path(name).stem
            positions = evaluate_positions(root, data_dir, phase=phase, split=evaluation_name)
            lines += ['正文位置诊断（相对每条消息完整正文，单独于整段输入位置）：', '',
                      '| 正文位置 | 评分位置数 | Accuracy | Macro-F1 |', '|---|---:|---:|---:|']
            for group, score in {**positions['by_body_quartile'], **positions['by_body_edge']}.items():
                lines.append(f"| {group} | {score['tokens']} | {number(score['accuracy'])} | {number(score['macro_f1'])} |")
            lines += ['', f"按角色的尾部 recall/误报、正文长度与上下文策略分项见 `{evaluation_name}-positions.json`；不存在的类别仍为 N/A。", '']
    frozen = artifacts.get('freeze.json')
    if frozen:
        lines += [f"冻结分类器 SHA256：`{frozen['probe_sha256']}`。六类顺序、标准化器、选头顺序、模型/tokenizer 和特征位置随导出绑定，旧五类文件由新消费端拒绝。", '']
        lines += ['消费端短例（在运行机的已固定依赖环境中执行，无需加载 120B）：', '', '```python',
                  'import sys', 'from pathlib import Path', 'import numpy as np',
                  f'sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})',
                  'from synthetic_agent_classifier import FeatureBank, ROLES, load_probe', '',
                  f'run = Path({str(root.resolve())!r})',
                  'probe = load_probe(run / "best.npz")',
                  f'bank = FeatureBank(run / "features" / {phase!r}, "dev", expected_binding=probe.metadata)',
                  'x = bank.read(np.arange(min(8, len(bank))))  # 原始 N×1024 特征',
                  'probabilities = probe.predict_proba(x)      # N×6，自动应用 FIT 标准化',
                  'assert probabilities.shape == (len(x), 6)',
                  'assert tuple(probe.metadata["roles"]) == ROLES',
                  'print(ROLES, probe.metadata["heads"], probabilities)', '```', '',
                  '六类固定顺序为 system、developer、user、cot、assistant、tool。输入列必须是同一冻结模型/tokenizer 的 pre-o_proj 信号，按 metadata.heads 的固定顺序拼接 16×64 维；不得另行排序头或重复标准化。FeatureBank 的 expected_binding 会核对 model_binding、选头与来源绑定；仅有 N×1024 形状并不能证明任意向量与模型匹配。', '']
    recovery = artifacts.get('resume-validation.json')
    if recovery:
        lines += [f"实际 GPU 恢复验证：{recovery.get('passed')}；权重最大差异 {recovery.get('weight_max_absolute_error')}；损失差异 {recovery.get('loss_absolute_error')}；NPZ 概率最大差异 {recovery.get('numpy_export_max_absolute_error')}。", '']
    lines += ['C 标签表示 analysis channel 中的文章正文，不能解读为真实推理语义。120B 本体冻结，本报告只评估旁路分类器；本文不声称预训练大模型未接触过对话。', '',
              '外部测试流程规定仅在冻结后打开 OASST/ToxicChat 各 128 条，是否已执行见本报告已有评估及缺失清单；新增 S/D 是固定事实段落，属于后加上下文，正式外部指标只评分原始 U/A。历史对话原文留在本地，服务器只接收去重 SHA256 和公开来源 ID。', '',
              '历史 CoT 清理版本有相同事件正文的上下文对照；工具调用前后阶段含不同目标正文，其整体分数不能解释为删除 CoT 的因果效果。统一 user/去标签图仅为原角色语义诊断，不作为改变格式后的形式角色 accuracy。', '']
    for plot_index in sorted((root / 'plots').glob('*/index.json')) if (root / 'plots').exists() else []:
        lines += [f"图表清单：[{plot_index.parent.name}]({plot_index.relative_to(root)})。", '']
    lines += ['未完成或缺少的产物：' + ('、'.join(result['missing']) or '无'), '']
    (root / 'REPORT.zh-CN.md').write_text('\n'.join(lines))
    (root / 'report-status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
