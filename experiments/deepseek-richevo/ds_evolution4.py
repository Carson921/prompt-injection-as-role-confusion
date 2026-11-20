"""Fourth DeepSeek evolution: corrected role probe (GPT-OSS developer -> system, GPT-OSS system dropped),
one benchmark-specific objective per run (DS_TARGET=agentdojo | ipi).

Same search loop, vocabulary and cached-prefix surrogate as ds_evolution2/3.  All "measured facts" given to
the proposal LLM are replaced by the text in DS_FACTS (a file written from measurements made with the
corrected probe), because the numbers quoted by the earlier runs came from the mislabeled probe.
"""
from __future__ import annotations

import os
from pathlib import Path

import ds_evolution2 as E2

R, L = E2.R, E2.L
TARGET = os.environ['DS_TARGET']
E2.PromotingFastScorer.STEP = float(os.environ.get('DS_EXACT_STEP', '0.03'))
E2.PromotingFastScorer.FLOOR = float(os.environ.get('DS_EXACT_FLOOR', '0'))
L.SYSTEM = L.SYSTEM[:L.SYSTEM.index('\n\nMeasured facts from this study')] + '\n\n' + Path(os.environ['DS_FACTS']).read_text().strip() + '\n'

if __name__ == '__main__':
    R.main()
