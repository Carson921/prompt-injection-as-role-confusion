"""Export an evidence-based comparison of the completed local 20B/120B demos."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent / 'outputs'
OUT20 = ROOT / 'role-probe-a100'
OUT120 = ROOT / 'role-probe-120b-a100'


def main():
    m20 = json.loads((OUT20 / 'metrics.json').read_text())
    m120 = json.loads((OUT120 / 'metrics.json').read_text())
    memory = json.loads((OUT120 / 'memory_stats.json').read_text())
    p20 = pd.DataFrame(m20['accuracy_by_layer']).assign(model='20B', relative_depth=lambda d: d.layer / 24)
    p120 = pd.DataFrame(m120['accuracy_by_layer']).assign(model='120B', relative_depth=lambda d: d.layer / 36)
    all_accuracy = pd.concat([p20, p120], ignore_index=True)
    all_accuracy.to_csv(OUT120 / 'accuracy_comparison.csv', index=False)
    user_frames = []
    for name, folder in [('20B', OUT20), ('120B', OUT120)]:
        frame = pd.read_csv(folder / 'token_role_projections.csv', keep_default_na=False)
        frame = frame[(frame.target_role == 'cot') & (frame.role == 'user')].sort_values('sample_ix').copy()
        frame['user_token_index'] = range(len(frame))
        frame['model'] = name
        user_frames.append(frame)
    assert user_frames[0].token.tolist() == user_frames[1].token.tolist(), 'User prompts differ'
    shared = pd.concat(user_frames, ignore_index=True)
    shared.to_csv(OUT120 / 'user_cotness_comparison.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for frame, name in [(p20, '20B (BF16/eager)'), (p120, '120B (MXFP4/Triton)')]:
        axes[0].plot(frame.relative_depth, frame.accuracy, 'o-', label=name)
    axes[0].set(xlabel='Layer index / number of layers', ylabel='Held-out token accuracy', ylim=(0, 1.02), title='Role probes: original prompt split')
    axes[0].legend(fontsize=8)
    for frame, name in zip(user_frames, ['20B, layer 12', '120B, layer 18']):
        axes[1].plot(frame.user_token_index, frame.prob, '.-', markersize=3, linewidth=.8, label=name)
    axes[1].axvspan(22, len(user_frames[0]) - 1, alpha=.08, color='green', label='Forged reasoning inside user')
    axes[1].set(xlabel='Token index within the identical user message', ylabel='P(cot)', ylim=(-.03, 1.03), title='Each model\'s middle-layer probe')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT120 / 'comparison.png', dpi=160)
    plt.close(fig)

    interactive = make_subplots(rows=1, cols=2, subplot_titles=['Probe accuracy by relative layer position', 'CoT probability within the identical user message'])
    for frame, name in [(p20, '20B'), (p120, '120B')]:
        interactive.add_trace(go.Scatter(x=frame.relative_depth, y=frame.accuracy, text=frame.layer.astype(str), mode='lines+markers', name=name), row=1, col=1)
    for frame, name in zip(user_frames, ['20B layer 12', '120B layer 18']):
        interactive.add_trace(go.Scatter(x=frame.user_token_index, y=frame.prob, text=frame.token, mode='lines+markers', name=name, marker={'size':4}, hovertemplate='Token %{x}: %{text}<br>P(cot)=%{y}<extra>%{fullData.name}</extra>'), row=1, col=2)
    interactive.update_xaxes(title_text='Layer index / layer count', row=1, col=1)
    interactive.update_xaxes(title_text='User token index', row=1, col=2)
    interactive.update_yaxes(range=[-.03, 1.03])
    interactive.write_html(OUT120 / 'comparison.html', include_plotlyjs=True)

    a20 = {r['layer']:r['accuracy'] for r in m20['accuracy_by_layer']}
    a120 = {r['layer']:r['accuracy'] for r in m120['accuracy_by_layer']}
    layer_rows = '\n'.join(f"  第 {layer:2d} 层: 20B {a20[layer]:.4%}; 120B {a120[layer]:.4%}" for layer in sorted(a20.keys() & a120.keys()))
    extra_rows = '\n'.join(f"  第 {layer:2d} 层: {a120[layer]:.4%}" for layer in sorted(a120.keys() - a20.keys()))
    reg20 = {r['region']:r for r in m20['cotness_by_region']}
    reg120 = {r['region']:r for r in m120['cotness_by_region']}
    report = f'''GPT-OSS-120B/A100 角色探针实验报告

完整运行已完成。模型位于 /workspace/gpt-oss-120b，沿用项目的独立 uv venv 和 GPU 0（A100 80GB）。

显存结论
单卡可以完成本次 MXFP4 实验。NVML 每秒采样峰值约 {memory['nvml_peak_used_gib']:.2f} GiB，总显存 {memory['nvml_total_gib']:.2f} GiB。
PyTorch 峰值分配 {m120['torch_peak_allocated_gib']:.2f} GiB，峰值预留 {m120['torch_peak_reserved_gib']:.2f} GiB。
仅本地 MXFP4 权重文件约 60.77 GiB；若像 20B 那样全量展开为 BF16，仅专家权重就约 213.57 GiB，因此无法全量放入这张卡。
本次保留 MXFP4，使用固定版本的 Triton 专家内核、BF16 激活、eager attention、batch_size=2、正文最大 512 token。
该结论适用于本次短序列和小 batch，不能直接推广为任意上下文或并发规模。

实验设置
复用 20B 运行缓存的同一批 150 条文本（C4/Dolma3 各75）、同一个缓存伪造文本和问题。
两模型 tokenizer.json 的 SHA256 完全相同。
五种角色包装共 750 序列；使用 system/user/cot/assistant 四类探针。
保留原 demo 的 seed=123、prompt_ix 90/10 划分、L2 logistic regression、C=0.005、max_iter=2000。
120B 为 36 层，因此按每四层取样并增加中间层18：{m120['layers']}。
20B 的示例投影层是12，120B 使用18，各为模型的中间位置；层编号从0开始。

共同绝对层的测试准确率（层号相同不代表相对深度相同）
{layer_rows}

120B 额外层的测试准确率
{extra_rows}

同一 user 消息中的平均 CoT 概率
20B 第12层：普通文本 {reg20['ordinary_user_text']['mean_cot_probability']:.4%}；伪造推理 {reg20['forged_cot_in_user']['mean_cot_probability']:.4%}。
120B 第18层：普通文本 {reg120['ordinary_user_text']['mean_cot_probability']:.4%}；伪造推理 {reg120['forged_cot_in_user']['mean_cot_probability']:.4%}。
120B 第18层双精度探针复核测试准确率：{m120['solver_diagnostic']['test_accuracy']:.4%}。
120B 本地生成 {m120['generation']['generated_tokens']} 个 token，耗时 {m120['generation']['seconds']:.2f} 秒，正常 EOS 结束：{m120['generation']['ended_with_eos']}。
全流程耗时约 {memory['wall_seconds'] / 60:.2f} 分钟（含模型加载、首次 Triton 编译、激活导出和训练）。未调用付费 API。

解释边界
20B 使用 BF16/eager 专家后端，120B 使用 MXFP4/Triton 专家后端；因此这并非仅模型规模变化的严格算术后端控制实验。
原 demo 按角色包装后的 prompt_ix 划分，同一原始文本的不同角色版本可能跨越训练和测试集。
概率是线性探针的分类输出，不能直接视作攻击成功率；示例也不是完整论文评测。
具体收敛警告保留在 run.log 和已执行 notebook 中；原始单精度指标完整保留，附带中间层双精度复核。

产物
role-probe-demo.executed.ipynb：完整执行副本
metrics.json、memory_stats.json：运行与显存指标
probe_accuracy.csv、accuracy_comparison.csv：探针准确率及模型对比
cotness.html、comparison.html：单模型与对比交互图
comparison.png：对比静态图
probes.pkl：10层原始训练探针
middle_probe_float64.pkl、solver_diagnostic.json：第18层双精度复核
training_hidden_states.pt、training_tokens.parquet：激活及 token 对应关系，用于避免重复导出

再次运行
/root/prompt-injection-as-role-confusion/.venv/bin/python /root/prompt-injection-as-role-confusion/demo/run_role_probe_120b_a100.py
从已导出的激活恢复
/root/prompt-injection-as-role-confusion/.venv/bin/python /root/prompt-injection-as-role-confusion/demo/run_role_probe_120b_a100.py --resume-activations

MXFP4 运行要求参考：https://huggingface.co/docs/transformers/quantization/mxfp4
'''
    (OUT120 / 'report.txt').write_text(report)
    print(report[:2800])


if __name__ == '__main__':
    main()
