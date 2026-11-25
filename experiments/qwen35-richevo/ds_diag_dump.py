"""Diagnostic: all-head features for a sample of probe-training records and for real Qwen AgentDojo conversations."""
import json, random, sys
from pathlib import Path
import numpy as np
from ds_common import PROBE_RUN, ROLES, read_json
from ds_probe_client import Builder, call
OUT = Path('/scratch/wz55191/qwen-diag'); rng = random.Random(7)

def run(name, ids, positions, labels):
    path = OUT / f'{name}.npy'
    if not path.exists():
        call('dump', input_ids=ids, positions=positions, path=str(path))
        (OUT / f'{name}.json').write_text(json.dumps(dict(positions=positions, labels=labels)))
        print(name, len(positions), flush=True)

if sys.argv[1] == 'train':
    groups = {}
    for line in open(PROBE_RUN / 'data/inputs-fit.jsonl'):
        r = json.loads(line)
        key = r['structure'] if r['structure'] != 'c4_dolma3:single' else 'single:' + r['source_role']
        if r['domain'] != 'c4_dolma3': key = 'agentdojo'
        if r['input_tokens'] <= 4000: groups.setdefault(key, []).append(r)
    print({k: len(v) for k, v in groups.items()})
    quota = {'c4_dolma3:multi': 40, 'agentdojo': 30}
    for key, rows in sorted(groups.items()):
        for n, r in enumerate(rng.sample(rows, min(len(rows), quota.get(key, 8)))):
            pick = sorted(rng.sample(range(len(r['positions'])), min(32, len(r['positions']))))
            run(f"train-{key.replace(':', '_')}-{n}", r['input_ids'], [r['positions'][i] for i in pick], [r['labels'][i] for i in pick])
else:
    b = Builder(read_json('outputs/qwevo-20261120/agentdojo/dataset.json')); tok = b.codec
    from ds_refs4 import CONDITIONS
    for c, case in enumerate(b.cases()):
        for name in ('original', 'true_user_turn'):
            req = b.render_true_user(case) if name == 'true_user_turn' else b.render(case, CONDITIONS[name])
            ids = req['input_ids'][:max(req['goal_token_indices']) + 1]
            text = [tok.decode([i]) for i in ids]
            labels, role, i, goal = [None] * len(ids), None, 0, set(req['goal_token_indices'])
            while i < len(ids):
                t = text[i]
                if t == '<|im_start|>':
                    role = {'system': 'system', 'user': 'user', 'assistant': 'assistant'}[text[i + 1]]; i += 3
                    if role == 'user' and text[i] == '<tool_response>': role = 'tool'; i += 2
                    continue
                if t == '<think>': role = 'cot'; i += 2; continue
                if t == '</think>': role = 'assistant'; i += 2; continue
                if t in ('<|im_end|>', '</tool_response>'): role = None
                if t == '<tool_call>': role = None
                if role and t.strip(): labels[i] = role
                i += 1
            by = {}
            for i, l in enumerate(labels):
                if l: by.setdefault(l, []).append(i)
            pos = []
            for l, idx in by.items():
                if l == 'user':   # keep turns apart: first user turn vs later (the goal as a genuine turn)
                    late = [i for i in idx if i in goal] if name == 'true_user_turn' else []
                    early = [i for i in idx if i not in late]
                    pos += late + rng.sample(early, min(40, len(early)))
                else:
                    pos += rng.sample(idx, min(60, len(idx)))
            pos = sorted(set(pos))
            lab = [('user2' if (name == 'true_user_turn' and i in goal) else ('inj' if (name == 'original' and i in set(req['payload_token_indices'])) else labels[i])) for i in pos]
            run(f'real-{c}-{name}', ids, pos, lab)
