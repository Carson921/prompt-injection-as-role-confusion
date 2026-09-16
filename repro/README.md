# Reproduction: OpenRouter + gpt-oss-20b

Runnable ports of the paper's API-based experiments
([arXiv:2603.12277](https://arxiv.org/abs/2603.12277)), with two constraints
applied throughout:

* **every LLM call goes through OpenRouter** — no local model, no OpenAI API;
* **the only target model is `openai/gpt-oss-20b`.**
  `google/gemini-2.5-pro` is kept in the two auxiliary roles the paper gives it:
  author of the forged policies, and judge of the outcomes.

The notebooks under `experiments/` are the originals and are left untouched;
these scripts write to the same output paths the notebooks expect.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install pandas numpy aiohttp tqdm python-dotenv pyyaml datasets requests matplotlib
```

`.env` at the repo root needs `OPENROUTER_API_KEY` (and `HF_TOKEN` only if you
switch the StrongREJECT loader back to the gated HF mirror).

## What runs

### 1. CoT Forgery — chat jailbreaks (README §2.1)

| | script | replaces | output |
| --- | --- | --- | --- |
| 1 | `chat_01_generate_forgeries.py` | `cot-forgery-chat-evals/01` | `base-harmful-policies.csv` |
| 2 | `chat_02_run_generations.py` | `cot-forgery-chat-evals/03` | `openrouter-generations/harmful-responses-classified.csv` |
| 3 | `chat_03_plots.py` | `cot-forgery-chat-evals/04` (R) | `cot-forgery-chat-evals/plots/` |

```bash
.venv/bin/python repro/chat_01_generate_forgeries.py --batch-size 40
.venv/bin/python repro/chat_02_run_generations.py --n-sample 200
.venv/bin/python repro/chat_03_plots.py
```

### 2. CoT Forgery — agent prompt injection (README §2.2)

| | script | replaces | output |
| --- | --- | --- | --- |
| 1 | `agent_00_prepare_pages.py` | page-prep cells of `cot-forgery-agent-evals/01` | `scrapes/{raw,base-injection,cot-forgery-injection}` |
| 2 | `agent_01_run_injections.py` | ReAct + classify cells of `cot-forgery-agent-evals/01` | `agent-outputs-classified-openrouter-gptoss20.csv` |
| 3 | `agent_02_plots.py` | `cot-forgery-agent-evals/03` (R) | `cot-forgery-agent-evals/plots/` |

```bash
.venv/bin/python repro/agent_00_prepare_pages.py --n-pages 100
.venv/bin/python repro/agent_01_run_injections.py --n-sample 200 --workers 6
.venv/bin/python repro/agent_02_plots.py
```

## Sandboxing the agent eval

The agent under test gets a `bash` tool, and the whole point of the experiment is
that a web page talks it into exfiltrating a `.env` file. Running that tool
directly on the host — as the notebooks do — would expose this repo's real `.env`
and post it to a public paste site. `sandbox.py` + `sandbox_agent.py` put every
command the model emits inside an unprivileged bubblewrap namespace instead:

* **no network egress at all** — its own empty net namespace, loopback only;
* **no host filesystem** beyond read-only `/usr`, `/bin`, `/lib` and one scratch
  workspace holding a freshly generated throwaway `SECRET_API_KEY`;
* `0x0.st` and `dpaste.com` resolve to a local HTTPS stand-in that serves the
  injected page and accepts the exfiltration POST, so transcripts look exactly
  like the real thing and the success/failure label stays meaningful, while
  nothing leaves the machine.

`sandbox.py` run directly is a self-test of those properties.

## Deviations from the paper

| | paper | here | why |
| --- | --- | --- | --- |
| Target provider | `nebius/fp4` | `parasail/fp4` | Nebius no longer serves gpt-oss-20b on OpenRouter; Parasail is the closest match (fp4, tools + reasoning). |
| StrongREJECT | `walledai/StrongREJECT` (HF) | upstream `strongreject_dataset.csv` | The HF mirror is gated; the upstream CSV is the same 313 prompts in the same order. |
| Agent runtime | local gpt-oss-20b in Harmony format / OpenAI Responses API | OpenRouter chat completions with `tools` | No local GPU large enough; reasoning text is carried across turns so the loop stays equivalent. |
| Agent hosting | page uploaded to 0x0.st, exfil to dpaste.com | local HTTPS stand-in for both | Isolation (see above); also removes a dependency on third-party rate limits. |
| Page scrapes | writes the raw response bytes | writes the class-stripped HTML | The notebook checks the size limit against the stripped text but saves the unstripped bytes; saving the stripped text is what keeps pages under the stated 100KB. |
| Plots | R / ggplot2 | matplotlib | R is not installed here; `plotstyle.py` approximates `theme_iclr`. |
| Injection payload choice | `random.choice` with `random` unseeded | same, with `random.seed(1234)` | Reproducibility. |

## Not reproduced

Everything that needs hidden states from a locally-run gpt-oss-20b:
`role-analysis/02-04`, `cot-forgery-role-confusion/*`, `agent-injections/*`,
`position-analysis/*`. Role probes are trained on activations, which no hosted
API exposes. gpt-oss-20b needs an sm_90-class GPU for its MXFP4 kernels, or
~40GB of VRAM once dequantized to bf16; this machine has an RTX 2080 Ti (11GB,
sm_75) and a Tesla P4 (8GB, sm_61).

## Cost and caching

Every OpenRouter call is cached under `repro/cache/`, tagged with a hash of the
prompt that produced it, so an interrupted run resumes instead of re-paying and a
changed prompt set invalidates only the entries that no longer apply. Agent
transcripts append to `cot-forgery-agent-evals/openrouter-agent-runs.jsonl` and
resume by page path.

## Rate limits

OpenRouter caps new accounts at **20 requests/minute for `google/gemini-2.5-pro`**,
which is the limiter for every step that uses the auxiliary model. `run_openrouter`
paces itself to 18 rpm, but that budget is per process — **do not run two
gemini-backed steps at once** (e.g. `chat_01` and `agent_01`'s classification),
or both will spend their time in 429 backoff. Run them one after another.
