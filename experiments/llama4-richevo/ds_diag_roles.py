import sys, numpy as np
from ds_common import ROLES, read_json
from ds_probe_client import Builder, call
from ds_refs4 import CONDITIONS
b = Builder(read_json(sys.argv[1])); case = b.cases()[int(sys.argv[2])]
for name in sys.argv[3:]:
    req = b.render_true_user(case) if name == 'true_user_turn' else b.render(case, CONDITIONS[name])
    ids = req['input_ids'][:max(req['goal_token_indices']) + 1]
    start = max(0, len(ids) - 900)
    pos = list(range(start, len(ids)))
    p = np.asarray(call('full', input_ids=ids, positions=pos)[0]['probabilities']); lab = p.argmax(1)
    print('=====', name, case['case_id'], len(ids), 'goal', req['goal_token_indices'][0], req['goal_token_indices'][-1])
    i = 0
    while i < len(pos):
        j = i
        while j < len(pos) and lab[j] == lab[i]: j += 1
        text = b.codec.decode(ids[pos[i]:pos[j - 1] + 1])
        print(f'[{ROLES[lab[i]]:9s} {pos[i]}-{pos[j-1]} p={p[i:j, lab[i]].mean():.2f}]', repr(text[:70] + (' ... ' + text[-60:] if len(text) > 140 else text[70:])))
        i = j
