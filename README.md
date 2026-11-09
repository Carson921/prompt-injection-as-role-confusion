
<h1 align="center">Prompt Injection as Role Confusion</h1>
<p align="center">
  <a href="https://role-confusion.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-1d91c0?style=flat-square"></a>
  <a href="https://arxiv.org/pdf/2603.12277"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper%20PDF-e89522?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2603.12277"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-b31b1b?style=flat-square"></a>
  <a href="https://github.com/role-confusion/prompt-injection-as-role-confusion"><img alt="Code" src="https://img.shields.io/badge/Code-24292f?style=flat-square"></a>
</p>

## Overview

LLMs see the world as a single stream of text, partitioned into *roles* like `<user>` or `<tool>`. We trace **prompt injection** to **role confusion**: models perceive the source of text from *how it sounds*, not its labeled role. A command hidden in a webpage hijacks an agent simply because it sounds like `<user>` text, despite its `<tool>` label.

This repo provides:

- **Role probes**: train and apply linear probes to measure how models internally perceive "who is speaking"
- **CoT Forgery**: reproduce the zero-shot attack that spoofs model reasoning, in both chat and agent settings
- **Role-space analysis**: project prompt injections into role space to visualize and quantify role confusion
- **Paper reproduction**: regenerate all plots and tables from the paper

The [quickstart demos](#quickstart) are the fastest way to get started; the [full reproduction notebooks](#full-reproduction) cover every experiment in the paper.

## ⚡Quickstart

We recommend starting here rather than cloning the full repo.

**Role probes demo** - train probes and test role confusion on real examples:
- Download and run `demo/role-probe-demo.ipynb` with `demo/simple_test_helpers.py` in the same directory

**CoT Forgery demo** - run the attack on a few examples:
- Download and run `demo/cot-forgery-demo.ipynb`

**Recorded local A100 runs** - execute the role-probe demo with models in
`/workspace/gpt-oss-20b` and `/workspace/gpt-oss-120b`:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r demo/requirements-role-probe-120b-a100.txt
.venv/bin/python -m ipykernel install --user --name role-probe-uv --display-name 'Role probe (project uv .venv)'
.venv/bin/python demo/run_role_probe_a100.py
.venv/bin/python demo/run_role_probe_120b_a100.py
.venv/bin/python demo/summarize_role_probe_runs.py
```

Both runs use the same 150 cached training texts and matched cached forgery,
with local generation and no model API calls. The 20B run uses BF16/eager
experts; the 120B run keeps MXFP4/Triton experts to fit an A100 80GB, with a
sampled whole-GPU peak of 76.19 GiB at batch size 2. This backend difference
must be considered when comparing the models.

The [20B report](demo/outputs/role-probe-a100/report.txt),
[120B report](demo/outputs/role-probe-120b-a100/report.txt), and
[comparison plot](demo/outputs/role-probe-120b-a100/comparison.png) include
results and limitations. Executed notebooks, trained probes, token projections,
and small plots are committed alongside the reports. Model files, the virtual
environment, and multi-gigabyte activation checkpoints stay local. These runs
are demo results, not the full paper evaluation.

All below instructions are only needed for full experiment-by-experiment replication of the paper.

## 🔨Requirements

### Hardware

The full experiments assume a CUDA GPU. The paper experiments were originally run on an H200. Smaller local experiments may work on other CUDA GPUs, but larger models activation-export workflows are memory-intensive and may require similarly high VRAM.

Activation exports and model outputs can also require substantial disk space. For a quick first pass, start with the `demo/` notebooks before running the full reproduction workflows.

### Software

- Python 3.12+
- CUDA 12.8
- R, optional, for some analysis and plotting notebooks

Install dependencies:

```bash
bash setup_python.sh # Change paths in file to local venv
bash setup_r.sh
```

Create a `.env` file in the repository root. Depending on which notebooks you run, you may need:

```bash
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...
```

API-based experiments can incur provider costs.

## 🔁Full Reproduction

The notebooks below are organized by experiment family. Within each family, run
the notebooks in order unless the step is marked optional.

1. [Role Space Analysis](#1-role-space-analysis)
2. [CoT Forgery Evaluations](#2-cot-forgery-evaluations)
3. [Role Confusion: CoT Forgery](#3-role-confusion-cot-forgery)
4. [Role Confusion: Standard Prompt Injections](#4-role-confusion-standard-prompt-injections)



## 1. Role Space Analysis

This workflow analyzes models' internal role perception. Notebooks and outputs
are model-specific; set the model choice in the relevant notebook.

Supported models include:

- `gpt-oss-20b`
- `gpt-oss-120b`
- `Nemotron-3-Nano`
- `Qwen3-30B-A3B`
- `Jamba-Reasoning-3B`

<p align="center">
  <img src="docs/cotness-phase-portrait-alt-tags.png" width="80%" alt="Role-space phase portrait">
</p>

| Step | Notebook | Requires | Main outputs |
| --- | --- | --- | --- |
| Generate conversational data | `role-analysis/01-get-conversations-data.ipynb` | OpenRouter or local model access | `convs/{model_name}.csv` |
| Train and evaluate role probes | `role-analysis/02-train-role-probes.ipynb` | Conversation data | `outputs/probes/{model_name}.pkl`, `outputs/probe-training/*.csv`, `outputs/probe-projections/*.csv` |
| Plot conversation projections | `role-analysis/03-analyze-probes.ipynb` | Trained probes and projections | `role-analysis/plots/*` |
| Plot gardening-conversation projections | `role-analysis/04-tomato-probe-results.ipynb` | Trained probes and projections | `role-analysis/plots/*` |

The first notebook builds model-specific conversation data from `toxicchat` and
`oasst`, regenerating model responses through OpenRouter or a local fallback.
The second notebook trains role probes on tag-induced role geometry and projects
both ordinary conversations and controlled examples into role space.

For the local GPT-OSS-120B MXFP4 follow-up, see the
[Chinese training guide](experiments/role-analysis/TRAINING-GUIDE-120B.zh-CN.md).
It explains the original method, the subsequent classifier changes, the archived
training dependencies, and the independently confirmed results with original and
new figures.

For six-label single-message activation collection on GPT-OSS-120B (1,000 C4
texts and 1,000 Dolma3 texts, all 36 pre-MLP layers), see the
[collection guide](experiments/single-message-pre-mlp/README.zh-CN.md).

## 2. CoT Forgery Evaluations

This workflow runs CoT Forgery in both chat and agent settings. The chat
notebooks evaluate jailbreak-style prompts; the agent notebooks evaluate
prompt injection in a ReAct-style tool loop.

### 2.1 Chat Jailbreaks

The chat notebooks generate CoT Forgery prompts, run local and closed-weight
model generations, classify attack success, and plot the resulting attack
success rates.

<p align="center">
  <img src="docs/user-eval-result.png" width="70%" alt="Chat evaluation results">
</p>

| Step | Notebook | Requires | Main outputs |
| --- | --- | --- | --- |
| Generate CoT Forgery prompts | `cot-forgery-chat-evals/01-generate-forgeries.ipynb` | `OPENROUTER_API_KEY` | `base-harmful-policies.csv` |
| Run local-model generations | `cot-forgery-chat-evals/02-export-jailbreak-generations.ipynb` | Generated prompts, local model access | `base-harmful-responses-classified.csv` |
| Run closed-model generations | `cot-forgery-chat-evals/03-run-openrouter-generations.ipynb` | Generated prompts, `OPENROUTER_API_KEY` | `openrouter-generations/harmful-responses-classified.csv` |
| Plot results | `cot-forgery-chat-evals/04-plot-jailbreak-stats.ipynb` | Classified generation outputs | `cot-forgery-chat-evals/plots/*` |

The first notebook calls an LLM through OpenRouter to generate CoT Forgery
prompts and comparison baselines for harmful questions in StrongREJECT. The
generation notebooks evaluate those prompts on local `gpt-oss-*` models and
closed-weight models. The classification step uses an LLM classifier through
OpenRouter.

### 2.2 Agent Prompt Injections

The agent notebooks evaluate CoT Forgery-style prompt injection in a ReAct-style
tool-use loop.

<p align="center">
  <img src="docs/agent-eval-result.png" width="70%" alt="Agent evaluation results">
</p>

| Step | Notebook | Requires | Main outputs |
| --- | --- | --- | --- |
| Run local-agent evaluations | `cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb` | Local `gpt-oss-*` model access | `local-agent-outputs-{model_name}-classified.csv` |
| Run hosted-agent evaluations | `cot-forgery-agent-evals/02-run-injections-openai.ipynb` | Hosted model API access | `api-agents-output-classified.csv` |
| Plot results | `cot-forgery-agent-evals/03-plot-injections.ipynb` | Classified agent outputs | `cot-forgery-agent-evals/plots/*` |

The outputs include full agent-loop transcripts and final success
classifications.

## 3. Role Confusion: CoT Forgery

This workflow uses the probes trained in the role-space workflow to analyze the
CoT Forgery activations from the chat and agent experiments, showing how styling/destyling impacts role confusion and how role confusion predicts attack success.

<p align="center">
  <img src="docs/cotness-redteam.png" width="90%" alt="CoTness analysis for red-team prompts">
</p>

| Step | Notebook | Requires | Main outputs |
| --- | --- | --- | --- |
| Export chat-attack activations | `cot-forgery-role-confusion/02-export-user-injection-activations.ipynb` | `cot-forgery-chat-evals/02-export-jailbreak-generations.ipynb` | `activations-redteam/{model_name}` |
| Export agent-attack activations | `cot-forgery-role-confusion/01-export-agent-activations.ipynb` | `cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb` | `activations-agent/{model_name}` |
| Project attacks into role space | `cot-forgery-role-confusion/03-project-role-probes.ipynb` | Trained probes and exported activations | `cot-forgery-role-confusion/exports/*` |
| Plot chat and agent role analyses | `cot-forgery-role-confusion/04-plot-injection-probe-results.ipynb`, `cot-forgery-role-confusion/05-plot-agent-probe-results.ipynb` | Role-projection exports | `cot-forgery-role-confusion/plots/*` |

If you only care about chat attacks, you can skip the agent activation export
and the agent plotting notebook. The projection notebook also has an
agent-analysis section that can be skipped when agent activations are not
available.

## 4. Role Confusion: Standard Prompt Injections

This workflow analyzes how role perception predicts success for standard agent
prompt injections, rather than CoT Forgery. This corresponds to the standard
prompt-injection role-confusion analysis in the paper, where we vary Userness exogenously and show role confusion predicts attack success in canonical agent prompt injections.

<p align="center">
  <img src="docs/userness-x-asr.png" width="60%" alt="Userness versus attack success rate">
</p>

| Step | Notebook | Requires | Main outputs |
| --- | --- | --- | --- |
| Run prompt-injection variants and project Userness | `agent-injections/01-run-user-injections-gpt-oss.ipynb` | Trained role probes | `outputs/agent-outputs-classified-{model_name}.csv` |
| Plot results | `agent-injections/02-analyze-injections.ipynb` | Classified outputs with Userness | `outputs/plots/*` |

The first notebook creates prompt-injection variants, runs the ReAct loop for
`gpt-oss-*`, classifies agent harm level, and extracts the mean Userness of each
injected query.

### 4.1 Optimizing Userness directly (rich evolution on the role probe)

`experiments/autodan-role-probe/rich_evolution.py` searches injection wrapper
templates whose only objective is the frozen role probe's mean goal-token
P(user) over six native AgentDojo tool contexts, under two hard constraints: no
genuine Harmony special-token spelling or ID may appear in the payload, and no
repeated text blocks are allowed (the `diverse_policy` admission rule). The
search combines LLM-generated seeds and GPTFuzzer-style LLM operators
(OpenRouter, abstract templates only), a near-delimiter grammar with variable
message count, role order and inter-message content, Unicode noise, an adaptive
operator bandit and a MAP-Elites archive, starting at 128 wrapper tokens and
growing the cap only when the search stalls. Within 4,324 scored templates the
best reached P(user) = 0.998 (previous non-repetitive best: 0.42). Templates
selected from different P(user) bands were then evaluated on the full 389-task
AgentDojo population with the frozen fixed-candidate evaluator:

| P(user) band | P(user) | AgentDojo ASR | Utility |
| --- | --- | --- | --- |
| 0.99–1.00 | 0.998 | 82.5% (321/389) | 0.15 |
| 0.90–0.99 | 0.990 | 83.5% (323/387) | 0.22 |
| 0.80–0.90 | 0.900 | 85.6% (333/389) | 0.10 |
| 0.70–0.80 | 0.800 | 75.1% (292/389) | 0.38 |
| 0.60–0.70 | 0.700 | 71.1% (276/388) | 0.30 |
| 0.50–0.60 | 0.600 | 71.0% (276/389) | 0.35 |
| 0.40–0.50 | 0.500 | 46.3% (180/389) | 0.47 |
| 0.30–0.40 | 0.398 | 73.9% (286/387) | 0.36 |
| 0.20–0.30 | 0.300 | 44.2% (172/389) | 0.38 |
| 0.10–0.20 | 0.200 | 29.0% (113/389) | 0.55 |
| plain baseline | 0.000 | 3.1% (12/388) | – |

ASR rises with P(user) overall (Spearman rho = 0.88 over the ten bands) but is
not monotone template by template: one 0.40-band template reaches 73.9% while
the 0.50-band one reaches 46.3%, so the probe score ranks wrappers coarsely
rather than predicting ASR per template. Above P(user) ≈ 0.9 ASR saturates at
82–86%. Method details, provenance and the full tables are in
`experiments/autodan-role-probe/outputs/richevo-20261105/METHOD.zh-CN.md` and
`REPORT.zh-CN.md`; `rich_report.py` regenerates the report and figures from the
ledgers.


### 4.2 Transfer to IPI Arena (same templates, same probe, tool-check judge)

`experiments/ipi-arena-richevo/` evaluates the ten band-selected templates and
the plain baseline on the open-source IPI Arena benchmark
(`ipi_arena_os`, 41 behaviors). Goals are imperative restatements of each
behavior's deterministic `tool_call` judge (36 primary behaviors; two with a
`tool_judge` but no `tool_call` criterion are reported separately; three
without a `tool_judge` are skipped). The target is gpt-oss-120b served by vLLM
(OpenAI API, temperature 0, up to five steps), WorldSim is
`google/gemini-2.5-flash` via OpenRouter, and a break is decided only by the
repository's unchanged tool judge. The same frozen probe scores the goal
tokens of each initial prompt (token-identical to vLLM's `/tokenize` output).

| P(user) band (AgentDojo) | P(user) IPI Arena | IPI Arena ASR | AgentDojo ASR |
| --- | --- | --- | --- |
| 0.99–1.00 | 0.838 | 52.8% (19/36) | 82.5% |
| 0.90–0.99 | 0.754 | 52.8% (19/36) | 83.5% |
| 0.80–0.90 | 0.660 | 61.1% (22/36) | 85.6% |
| 0.70–0.80 | 0.353 | 44.4% (16/36) | 75.1% |
| 0.60–0.70 | 0.575 | 44.4% (16/36) | 71.1% |
| 0.50–0.60 | 0.579 | 44.4% (16/36) | 71.0% |
| 0.40–0.50 | 0.384 | 47.2% (17/36) | 46.3% |
| 0.30–0.40 | 0.399 | 50.0% (18/36) | 73.9% |
| 0.20–0.30 | 0.348 | 38.9% (14/36) | 44.2% |
| 0.10–0.20 | 0.066 | 25.0% (9/36) | 29.0% |
| plain baseline | 0.005 | 8.3% (3/36) | 3.1% |

P(user) transfers across benchmarks (Spearman 0.85 between the AgentDojo and
IPI Arena probe scores) and still tracks ASR on IPI Arena (Spearman 0.87
between IPI Arena P(user) and IPI Arena ASR over the eleven conditions),
although absolute ASR is lower than on AgentDojo and the per-template
ordering is again not monotone. Details, per-behavior results and the goal
list are in `experiments/ipi-arena-richevo/outputs/run-20261108/REPORT.zh-CN.md`.

### 4.3 Optimizing directly on IPI Arena (higher P(user), no higher ASR)

Running the same rich-evolution search with the IPI Arena probe score as the
objective (a 12-behavior panel, 6 h, 1,493 scored candidates, wrapper cap 128
tokens; `experiments/ipi-arena-richevo/ipi_evolution.py`) raises the full
36-behavior goal-token P(user) from 0.838 (best AgentDojo template) to
0.899; every one of the top-20 panel candidates scores 0.887–0.899.
Re-running the tool-check ASR with the top five of them next to the plain
baseline and the two strongest AgentDojo templates (one WorldSim session,
304 episodes, `outputs/asr-ipievo-20261108/`):

| Condition | IPI P(user) | IPI ASR |
|---|---|---|
| ipievo_1 … ipievo_5 (IPI-optimized) | 0.895–0.899 | 44.4–58.3% (16–21/36) |
| AgentDojo band 0.99–1.00 | 0.838 | 52.8% (19/36) |
| AgentDojo band 0.80–0.90 | 0.660 | 63.9% (23/36) |
| plain baseline | 0.005 | 11.1% (4/36) |

The extra probe mass does not buy more breaks: across the IPI-optimized
templates the remaining failures are mostly explicit refusals with no tool
call at all (9–11 of 15–20 failures per template), i.e. the model reads the
goal as the user's request and declines it on content grounds, which the
role probe cannot see. The eight-condition Spearman between P(user) and ASR
drops to −0.05, consistent with the P(user)–ASR relation saturating around
0.85–0.9 on this benchmark.

## Citation

```bibtex
@inproceedings{ye2026promptinjectionroleconfusion,
  title = {Prompt Injection as Role Confusion},
  author = {Ye, Charles and Cui, Jasmine and Hadfield-Menell, Dylan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026},
  url = {https://arxiv.org/abs/2603.12277}
}
```

## License

See `LICENSE` for license information.
