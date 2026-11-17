# DeepSeek V4 Flash 角色探针加载方法

以下用法适用于已经生成 `freeze.json` 和 `checkpoint.npz` 的完成运行。最终训练、TEST 与独立验证的完成状态见输出目录的 `independent-validation.json`；只有其 `status` 为 `passed` 才表示最终验收通过。

默认输出目录：

```text
/home/wz55191/Github/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/deepseek-v4-flash-0731-role-probe-20261115
```

该文件使用 `deepseek-role-probe-v1` 格式，请通过 `deepseek_role_probe_train.Probe` 加载。旧 GPT-OSS 的 `synthetic_agent_classifier.load_probe`、归档 ChatInject `RoleProbe` 及其固定头捕获器不支持这个格式。

## CPU 加载与预测

以下示例只读取已经提取的 FIT 特征，不加载大模型，也不使用 GPU。依赖 NumPy，以及 SciPy 或 PyTorch 中的任意一个；当前训练环境缺少 SciPy 时，消费者使用 PyTorch 的 CPU `erf` 计算精确 GELU。

```python
import json
from pathlib import Path
import sys

import numpy as np

repo = Path("/home/wz55191/Github/prompt-injection-as-role-confusion")
source = repo / "experiments/role-analysis"
sys.path.insert(0, str(source))

from deepseek_role_probe_train import Probe, verify_freeze

run = source / "outputs/deepseek-v4-flash-0731-role-probe-20261115"
frozen = verify_freeze(run)  # 校验 checkpoint / training / selection 的冻结哈希。
manifest = json.loads((run / "features/manifest.json").read_text())
probe = Probe(run / "checkpoint.npz", expected_binding=manifest)

entry = manifest["splits"]["fit"][0]
x = np.load(run / "features" / entry["path"] / "x.npy",
            mmap_mode="r", allow_pickle=False)
probabilities = probe.predict_proba(x[:8])

print("checkpoint SHA256:", frozen["probe_sha256"])
print("columns:", probe.roles)
print("ordered heads:", probe.metadata["heads"])
print("probability shape:", probabilities.shape)  # (8, 5)
print(probabilities)
predicted_roles = np.asarray(probe.roles)[probabilities.argmax(axis=1)]
print(predicted_roles)
```

`predict_proba` 自行且仅执行一次 FIT 标准化，随后执行已选 linear 或 MLP，并返回 FP32 softmax 概率。请传入未经标准化的原始特征。

## 输入与列顺序

- 输入为 `N × 8192`；训练缓存为 FP16。新提取的特征应保持同样的 FP16 舍入，再由消费者转换到 FP32。
- 严格按 `probe.metadata["heads"]` 的顺序拼接 16 个头，每头 512 维。全局头编号为 `layer × 64 + head`。
- 捕获点是逆 RoPE 后、`self_attn.o_a_proj` 前的注意力头输出，标记为 `attention-head-output-before-o_a_proj`。它不同于 residual hidden states，也不同于已经混合头的 `o_b_proj` 输入。
- 输出列顺序从 `probe.roles` 读取。本次原生五类为 `system, user, cot, assistant, tool`，其中 `developer` 按官方编码并入 `user`。
- 特征必须来自检查点绑定的 DeepSeek V4 Flash 权重、配置及编码器。输入文本应使用官方 DeepSeek 编码器，body 位置应按该 tokenizer 的 token 索引确定。

如果已有同一基座实例，可使用现有捕获器获得新输入的特征：

```python
from deepseek_role_probe_model import extract_selected

# model 是已按该运行协议加载的冻结基座；此步骤需要 GPU。
x = extract_selected(model, input_ids, body_token_indices,
                     selected=probe.metadata["heads"])
probabilities = probe.predict_proba(x)
```

## 检查点内容

`mean` / `scale` 均为长度 8192 的 FIT 统计；`metadata` 是无需 pickle 的 JSON 字符串，包含角色顺序、头顺序、模型绑定、选头哈希、维度、隐藏层大小及激活函数。

linear 的参数名为 `weight::0.weight` / `weight::0.bias`，形状分别为 `(5, 8192)` / `(5,)`。MLP 还包含 `weight::3.weight` / `weight::3.bias`；第一层形状为 `(hidden, 8192)`，输出层为 `(5, hidden)`，`hidden` 从 metadata 读取，不能预设最终胜出架构。

AgentDojo 指标来自已有 GPT-OSS 轨迹的跨模型 teacher-forced replay；它不等同于新的 DeepSeek agent rollout 或攻击成功率测量。五分类与原 GPT-OSS 六分类的指标比较也应考虑标签空间和分词差异。
