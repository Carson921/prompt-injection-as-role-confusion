"""Compare the lockstep surrogate (one `chunks` call), the per-context `chunk` surrogate and exact full forwards."""
import sys, time
import numpy as np
from ds_fast_scorer import FastScorer
from ds_refs4 import CONDITIONS

dataset, out = sys.argv[1], sys.argv[2]
scorer = FastScorer(dataset, out)
for name in sys.argv[3:] or ['near_simple', 'near_close_user', 'ctrl_leadin']:
    template = CONDITIONS[name]
    t = time.time(); lock = scorer.score_cases(template); t_lock = time.time() - t
    t = time.time(); single = [scorer.score_case(c, template, exact=False) for c in scorer.cases]; t_single = time.time() - t
    t = time.time(); exact = [scorer.score_case(c, template, exact=True) for c in scorer.cases]; t_exact = time.time() - t
    a, b, e = (np.array([r['goal_user_mean'] for r in rows]) for rows in (lock, single, exact))
    print(name, 'means lock/chunk/exact', a.mean().round(4), b.mean().round(4), e.mean().round(4),
          'max|lock-chunk|', np.abs(a - b).max().round(4), 'max|lock-exact|', np.abs(a - e).max().round(4),
          'seconds', round(t_lock, 1), round(t_single, 1), round(t_exact, 1), flush=True)
    print('   per case exact', e.round(3).tolist(), flush=True)
    print('   per case lock ', a.round(3).tolist(), flush=True)
