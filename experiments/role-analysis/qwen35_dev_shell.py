"""Development shell: load the Qwen3.5 backbone once, then execute python files dropped into a directory.

usage: qwen35_dev_shell.py DIR   -> runs DIR/NAME.py (globals: model, backend, tokenizer, torch, np) and writes DIR/NAME.out
"""
import contextlib, io, sys, time, traceback
from pathlib import Path
import numpy as np
import qwen35_role_probe_model as backend

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
model = backend.load_model(lm_head=True)
import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(backend.DEFAULT_MODEL, local_files_only=True)
scope = dict(model=model, backend=backend, tokenizer=tokenizer, torch=torch, np=np)
print("READY", flush=True)
while True:
    for path in sorted(root.glob("*.py")):
        out = path.with_suffix(".out")
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            continue
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                exec(compile(path.read_text(), str(path), "exec"), scope)
            except Exception:
                traceback.print_exc(file=buffer)
        out.write_text(buffer.getvalue() + "\n[done]\n")
        torch.cuda.empty_cache()
    time.sleep(1)
