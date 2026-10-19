# Local gpt-oss-20b role-space reproduction

This runs locally adapted copies of all four notebooks on an A100 80 GB. The
original notebooks, role templates, and probe configuration remain the source of
the experimental settings. Haiku, MiniMax, and Kimi cross-model conversation
comparisons are skipped, as requested; no substitute data is fabricated.

## Environment

From the repository root, create a dedicated environment if it does not exist:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/role-analysis/requirements-local-a100.txt
.venv/bin/python -m ipykernel install --user --name role-probe-uv
apt-get update
apt-get install -y --no-install-recommends r-base-core r-cran-tidyverse r-cran-fs r-cran-ggtext r-cran-systemfonts r-cran-zoo r-cran-irkernel fonts-texgyre
Rscript -e 'IRkernel::installspec(user=TRUE, name="role-analysis-r", displayname="Role analysis (R)")'
```

The Python environment must be able to import the repository's `utils` package.
The runner adds the repository root to `PYTHONPATH` for its notebook kernels.
The local model is `/workspace/gpt-oss-20b`; weights and tokenizer are never
downloaded. Generation uses native MXFP4 experts with the Triton kernel revision
pinned in `local_runtime.py`, BF16 activations, and eager attention. Probe training
and validation use BF16-dequantized weights with eager experts and attention,
matching the earlier 20B demonstration's weight format. Probe training temporarily
moves model weights to CPU while fitting logistic regression to leave GPU memory
for the independent generation process.

## Run

```bash
.venv/bin/python experiments/role-analysis/prepare_local_data.py
.venv/bin/python experiments/role-analysis/validate_local_adaptations.py
.venv/bin/python experiments/role-analysis/run_local_gptoss20b.py --stage conversations
.venv/bin/python experiments/role-analysis/run_local_gptoss20b.py --stage training
.venv/bin/python experiments/role-analysis/run_local_gptoss20b.py --stage r-analysis
.venv/bin/python experiments/role-analysis/run_local_gptoss20b.py --stage r-tomato
.venv/bin/python experiments/role-analysis/summarize_local_run.py
```

Conversation generation and training can run concurrently on the same A100,
but process switching substantially slows eager expert operations. Sequential
execution is preferable for throughput. Training waits for the conversation
completion marker before starting validation if generation is still running.
For whole-device memory measurements, start `monitor_local_run.py` before those
stages, and create `outputs/local-gptoss20b/stop-monitor` after all stages finish.
Remove that marker before starting a new monitoring session.

## Settings and adaptations

- The configured 250 training texts yield 249 actual texts: `int(250*.25)` C4
  texts plus `int(250*.75)` Dolma3 texts, matching the original code.
- Texts are truncated to 1,024 tokens and rendered separately in five roles.
  Twelve layers (0, 2, ..., 22) and eight role combinations produce 96 probes.
  Logistic regression settings, role labels, and the 90/10 prompt split are
  unchanged. Copies of a base text in different roles can cross that split.
- Local conversations use 100 OASST and 100 ToxicChat candidates, at most two
  user turns, temperature 1, and a 4,000-token generation limit. Cached batches
  have deterministic seeds derived from their complete inputs and parameters.
  Generation uses an initial 16-conversation batch followed by batches of 64;
  long prefills and memory pressure trigger smaller batches automatically.
  Conversations with incomplete generations are removed in their entirety.
- The original local export retained only the last assistant segment and lost
  its CoT. Harmony `analysis` and `final` channels are extracted separately, and
  only final answers enter subsequent-turn history. Generation explicitly uses
  tokenizer EOS 200002 and PAD 199999, overriding the model's default multiple-EOS
  configuration. The parser retains the first EOS and strips later padding.
- Actual-model hooks capture the original pre-MLP location, immediately after
  `post_attention_layernorm`, and verify model logits. This avoids the original
  handwritten forward pass's incompatibility with the installed model backend.
- Activations stream into FP16 disk-backed maps. Training/validation batch sizes
  are reduced without changing input order. Maps are released after their
  derived probes or summaries have been saved. Allow at least 50 GiB of free
  local disk for the largest transient training map.
- CuPy's cache is capped at 12 GiB to avoid accumulating unused feature matrices
  from all eight role combinations. This changes memory reuse, not probe fitting.
- A separate float64 SUCA/layer-12 diagnostic checks optimizer sensitivity without
  replacing the original 96 float32 probes. The report also compares its tomato
  predictions with the main experiment. Supplemental executed notebooks record
  the memory guard and diagnostic applied during the live run.
- Validation uses the source notebook's full-test setting of up to 100 filtered
  conversations, rather than its 30-conversation preview. All four formats are
  evaluated: correct tags, no tags, all user tags, and all tool tags.
- All-layer projection statistics are aggregated per probe, preserving the
  source's per-conversation averaging and probability rounding. The equivalence
  check covers unequal conversation lengths and tied argmax probabilities.
- Tomato inputs use their actual maximum token length to prevent truncation.
  All three formats and all 96 probes are evaluated. Both R notebooks execute
  with IRkernel; their original smoothing and token alignment are retained.
  Empty strings from split Unicode tokens remain empty strings when read into R.
  Figure copies are stored with this run rather than overwriting `docs` images.
- The R analysis's original representative-layer expression selects layer 14;
  the Python single-probe example and tomato analysis select layer 12.

Executed notebooks, the text report, environment/provenance records, and detailed
run metadata live in `outputs/local-gptoss20b`. Canonical probe CSVs, projections,
and probe pickles are also written to the original notebook output directories.
Large transient caches are excluded from version control.
