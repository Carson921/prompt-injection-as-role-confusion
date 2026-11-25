"""Diagnostic: which full-attention layers / heads carry role information that transfers from the probe-training
records to real Qwen AgentDojo conversations (features from ds_diag_dump.py)."""
import glob, json, sys
import numpy as np
import torch
torch.set_num_threads(16)
class LogisticRegression:
    """Multinomial logistic regression with L2 (strength 1/C), full-batch L-BFGS."""
    def __init__(self, C=1.0, max_iter=300): self.C, self.max_iter = C, max_iter
    def fit(self, X, y):
        X, y = torch.as_tensor(X), torch.as_tensor(y)
        W = torch.zeros(X.shape[1] + 1, 5, requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=self.max_iter, history_size=20, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(X @ W[:-1] + W[-1], y, reduction='sum') + 0.5 / self.C * (W[:-1] ** 2).sum()
            loss.backward(); return loss
        opt.step(closure); self.W = W.detach(); return self
    def predict(self, X):
        X = torch.as_tensor(X); return (X @ self.W[:-1] + self.W[-1]).argmax(1).numpy()
D = '/scratch/wz55191/qwen-diag/'
ROLES = ['system', 'user', 'cot', 'assistant', 'tool']
def load(prefix):
    X, y, g = [], [], []
    for n, path in enumerate(sorted(glob.glob(D + prefix + '*.npy'))):
        meta = json.load(open(path[:-4] + '.json'))
        X.append(np.load(path).astype(np.float32)); y += meta['labels']; g += [path.split('/')[-1][:-4]] * len(meta['labels'])
    return np.concatenate(X), np.array(y), np.array(g)
Xt, yt, gt = load('train-'); Xr, yr, gr = load('real-')
yt = yt.astype(int)
real = {'user2': 1, 'inj': 4}; yi = np.array([real.get(l, ROLES.index(l) if l in ROLES else -1) for l in yr])
print('train', Xt.shape, np.bincount(yt), 'real', Xr.shape, {l: int((yr == l).sum()) for l in sorted(set(yr))})
records = sorted(set(gt)); held = np.isin(gt, records[::4])
def fit(A, B, C=0.1):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=300).fit((A - mu) / sd, yt[~held] if len(A) == (~held).sum() else yt)
    return m, lambda Z: m.predict((Z - mu) / sd)
def report(name, cols):
    A = cols(Xt); R = cols(Xr)
    m, pred = fit(A[~held], None)
    cv = (pred(A[held]) == yt[held]).mean()
    p = pred(R)
    parts = ' '.join(f"{l}={(p[yr == l] == yi[yr == l]).mean():.2f}" for l in ['system', 'user', 'cot', 'assistant', 'tool', 'inj', 'user2'] if (yr == l).any())
    print(f'{name:22s} heldout={cv:.3f} real={np.mean(p == yi):.3f}  {parts}', flush=True)
    return cv, np.mean(p == yi), p
for l in range(15):
    report(f'layer{3 + 4 * l:02d} (all heads)', lambda X, l=l: X[:, l].reshape(len(X), -1))
rows = []
for l in range(15):
    for h in range(32):
        A, R = Xt[:, l, h], Xr[:, l, h]
        m, pred = fit(A[~held], None, C=1.0)
        p = pred(R)
        rows.append((3 + 4 * l, h, (pred(A[held]) == yt[held]).mean(), (p == yi).mean(), (p[yr == 'user2'] == 1).mean(), (p[yr == 'tool'] == 4).mean(), (p[yr == 'cot'] == 2).mean()))
rows.sort(key=lambda r: -r[3])
print('top heads by real accuracy: (layer, head, heldout, real, user2, tool, cot)')
for r in rows[:40]: print('  ', r[0], r[1], *(f'{v:.3f}' for v in r[2:]))
json.dump(rows, open(D + 'heads.json', 'w'))
sel = [(3,3),(3,0),(7,27),(3,13),(7,28),(7,23),(7,21),(3,10),(7,16),(7,15),(3,5),(15,3),(59,26),(7,20),(19,7),(7,13)]
report('v2 selected 16', lambda X: np.concatenate([X[:, (a - 3) // 4, h] for a, h in sel], 1))
top = rows[:16]
report('top16 by real (optimistic)', lambda X: np.concatenate([X[:, (a - 3) // 4, h] for a, h, *_ in top], 1))
