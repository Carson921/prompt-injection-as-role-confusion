# Frozen six-role GPT-OSS-120B delivery

The final classifier is `formal/best.npz`. `formal/freeze.json` binds its exact SHA256,
six-role order, model/tokenizer, ordered sixteen heads, feature position, and FIT
standardizer. The frozen 120B backbone was not trained. Original articles were
placed in native message/channel wrappers; the C label is an operational analysis
channel label and does not assert genuine reasoning semantics.

Run `sha256sum -c SHA256SUMS` from this directory to verify the package. The
`bundle-manifest.json` records original remote paths, exact file hashes, protocol
bindings, and remote-retained large artifacts. Activation arrays, original epoch
histories, raw synthetic/external inputs, and full probability arrays remain on
the run host. Historical fitting and user-query CSV files remain on the original
local host; those CSV files were not uploaded. Their de-duplication contribution
on the run host consists of bound hashes and public source IDs. The separately
authorized public gardening diagnostic is archived outside this package.

The seven `curve-scalars.json` files are **derived summaries, not raw histories**.
They preserve every epoch's scalar losses/DEV criteria, class P/R/F1, sampling
counts, optimizer step, and improvement flag, and record each untouched raw
history's path/SHA. Seven candidate `complete.json` and `best.npz` files are also
included; final model selection used DEV only. TEST/external results and 2000
source-cluster confidence intervals appear under `formal/evaluations/`.

`formal/plots/test/` contains twenty native full-context masked tomato examples.
`formal/plots/dense/` contains fifteen fixed dense diagnostics: five templates by
native/all-user/no-tags. Changed-format scores refer to original semantic roles
and are not formal role accuracy. Numeric plot sidecars contain only positions,
offsets, role IDs, and probabilities, not raw source text. Figures intentionally
show the requested full visible context.

For CPU consumption, install NumPy/SciPy, then run:

```bash
python examples/load_probe.py /path/to/actual-selected-head-features.npy
```

Input columns must be the sixteen `metadata.heads` in fixed order, each 64
dimensions before o_proj, from the exact frozen model/tokenizer. A matching shape
alone does not establish feature provenance. Do not reorder heads or pre-apply
standardization. The full stage implementation and pinned requirements are in
`code/`; reproducing extraction needs the official model weights and four A40s.
The archived budget/deadline records are historical authorization limits, not a
fresh authorization to restart compute. GPU monitoring CSV/summary is delivered
separately by the task owner.
