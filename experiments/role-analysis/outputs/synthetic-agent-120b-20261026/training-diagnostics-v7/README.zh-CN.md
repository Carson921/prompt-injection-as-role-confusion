# 训练曲线可读性补充图

本目录是原始 `final-v7` 交付包之外的独立补充。原包、原 `training-curves.png/PDF`、配置和模型均未修改。

- 左图逐字使用导出字段 `train_loss_for_processed_batches`。它是该进程本轮实际处理批次的训练交叉熵记录；采用有放回的分层抽样，恢复训练时还可能只统计该进程新处理部分，因此不能称为全部唯一 FIT 样本的经验风险。
- 右图使用 `DEV structure_equal_macro_f1 × 100`，纵轴明确放大至 **99.35–99.88%**，不是从零起点。没有读取 TEST 或 external 指标。
- 七候选保持预声明顺序，显示全部 **152 个记录 epoch 点**。各曲线在实际最后一轮结束，没有补齐或外推到45轮。
- 两图的星号均来自已提交的 `best_epoch`，不是末轮，也不是按 TEST 或最低 DEV NLL 重选。蓝色粗线突出冻结 DEV winner `mlp128-seed17`：记录最佳第30轮，共训练38轮。

`curve-scalars.json` 是导出的数值摘要，不是原始 history。本目录分别保留摘要文件本身 SHA，以及原 history 的远端路径、字节数和原字节 SHA；原 history 没有被读取或修改。

文件：

- `training-loss-and-dev-zoom.png`、`.pdf`：补充图。
- `plot_training_diagnostics.py`：实际执行的原字节生成脚本。
- `source-sha256.json`：原包数值输入与原训练图的路径、字节数、SHA。
- `figure-manifest.json`：每条曲线的实际轮数、best、原 history 引用、坐标范围、脚本与图片 SHA。
- `visual-check.json`：实际 `view_image` 检查及回传字节校验。
- `SHA256SUMS`：本目录文件 SHA。

渲染使用 runpod 已有 Python 环境，`CUDA_VISIBLE_DEVICES=''`，OMP/OpenBLAS/MKL 各1线程，GNU timeout 上限120秒且不超过本轮授权绝对截止。没有安装依赖、加载模型、运行 forward 或启动 GPU。

在安装了 Matplotlib 的 CPU 环境中，可从本目录重现：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python plot_training_diagnostics.py --source-root ../final-v7 \
  --source-manifest source-sha256.json --output-dir .
```

实际查看已确认：完整横轴1–38、全部七条曲线与best标记、图例、明确的放大纵轴及脚注均可见，未发现剪裁。
