"""Qwen3.5 evolution with one benchmark-specific objective per run (DS_TARGET=agentdojo | ipi).

Same search loop as the DeepSeek dsevo4 runs (ds_evolution2 FastRun + PromotingFastScorer).  The "measured facts" given
to the proposal LLM come from the file DS_FACTS (written from reference measurements made with this model's probe).
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
