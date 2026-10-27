"""Use raw ordered pre-o_proj features from the bound frozen model; no toy data."""
import argparse
from pathlib import Path
import sys
import numpy as np
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "code"))
from synthetic_agent_classifier import load_probe, ROLES
p = argparse.ArgumentParser()
p.add_argument("features", type=Path, help="Actual raw N x 1024 selected-head features from the bound GPT-OSS model")
p.add_argument("--rows", type=int, default=8)
a = p.parse_args()
probe = load_probe(root / "formal" / "best.npz")
x = np.load(a.features, mmap_mode="r", allow_pickle=False)
probability = probe.predict_proba(x[:a.rows])  # FIT mean/scale are applied internally exactly once.
print("roles:", ROLES)
print("ordered heads:", probe.metadata["heads"])
print(probability)
