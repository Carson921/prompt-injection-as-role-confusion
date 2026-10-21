"""Independent numeric CPU check of natural-test paired component bootstrap."""
from pathlib import Path
import importlib
import itertools
import hashlib
import json
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0,'/root/prompt-injection-as-role-confusion/experiments/role-analysis')
r=importlib.import_module('systemness_natural_test_confirm')
source=Path(r.__file__);start=hashlib.sha256(source.read_bytes()).hexdigest()
checks=[]
def check(name,value=True):
    if not value:raise AssertionError(name)
    checks.append(name)

def compare_number(actual,expected):
    if expected is None:
        assert actual is None
        return 0.
    assert actual is not None and np.isfinite(actual) and abs(float(actual)-float(expected))<1e-12
    return abs(float(actual)-float(expected))

rows=[]
for mi,name in enumerate(r.NAMES):
    for i in range(4):
        s,u,a=10+i,20+i,15+i
        tp=0 if mi==0 and i<2 else min(s,6+i+mi)
        fp=0 if mi==0 and i<2 else i+mi
        sc=min(s,tp+2);uc=u-1-mi;ac=a-2
        row=dict(stratum='all-prepared',dataset='all',model=name,source_uid=f's{i}',family_id=f'f{max(i-1,0)}',
                 dependency_component_id='A' if i<2 else 'B' if i==2 else 'C')
        row.update(dict(zip(r.COUNT_FIELDS,[tp,fp,s-tp,u+a-fp,sc+uc+ac,s+u+a,sc,s,uc,u,ac,a])))
        rows.append(row)
counts=pd.DataFrame(rows)
comparisons,intervals,components=r.bootstrap(counts)
check('four models have2000 paired draws over three components',len(intervals)==4*9 and len(comparisons)==6*9 and len(components)==4*3)
ids=['A','B','C']
arrays={name:np.asarray([[sum(row[field] for row in rows if row['model']==name and row['dependency_component_id']==cid) for field in r.COUNT_FIELDS] for cid in ids],float) for name in r.NAMES}
# Independent formulas use explicit column offsets rather than production helper.
def independent(array):
    d=lambda a,b:np.divide(a,b,out=np.full_like(a,np.nan,dtype=float),where=b!=0)
    precision=d(array[...,0],array[...,0]+array[...,1])
    recall=d(array[...,0],array[...,0]+array[...,2])
    specificity=d(array[...,3],array[...,3]+array[...,1])
    return dict(system_precision=precision,system_recall=recall,
        system_f1=d(2*array[...,0],2*array[...,0]+array[...,1]+array[...,2]),
        system_balanced_accuracy=(recall+specificity)/2,system_false_positive_rate=d(array[...,1],array[...,1]+array[...,3]),
        accuracy=d(array[...,4],array[...,5]),role_recall_system=d(array[...,6],array[...,7]),
        role_recall_user=d(array[...,8],array[...,9]),role_recall_assistant=d(array[...,10],array[...,11]))
rng=np.random.RandomState(r.data.SEED+128432)
draw=rng.randint(3,size=(2000,3))
points={name:independent(array.sum(0)) for name,array in arrays.items()}
samples={name:independent(array[draw].sum(1)) for name,array in arrays.items()}
maximum=0.
for row in intervals:
    name,metric=row['model'],row['metric'];values=samples[name][metric];valid=np.isfinite(values)
    expected=float(points[name][metric]) if np.isfinite(points[name][metric]) else None
    maximum=max(maximum,compare_number(row['value'],expected))
    assert row['defined_bootstrap_replicates']==int(valid.sum())
    lo,hi=np.quantile(values[valid],[.025,.975])
    maximum=max(maximum,compare_number(row['ci95_low'],lo),compare_number(row['ci95_high'],hi))
for row in comparisons:
    before,after,metric=row['baseline_model'],row['comparison_model'],row['metric']
    difference=100*(samples[after][metric]-samples[before][metric]);valid=np.isfinite(difference)
    expected=100*(points[after][metric]-points[before][metric]);lo,hi=np.quantile(difference[valid],[.025,.975])
    maximum=max(maximum,compare_number(row['gain_percentage_points'],expected),compare_number(row['ci95_low'],lo),compare_number(row['ci95_high'],hi))
    assert row['defined_bootstrap_replicates']==int(valid.sum())
check('all point metrics/percentiles/6paired deltas match independent2000 draws')
residual=next(row for row in intervals if row['model']==r.NAMES[0] and row['metric']=='system_precision')
check('undefined precision draws are excluded and counted',0<residual['defined_bootstrap_replicates']<2000)
check('threshold system recall remains distinct from argmax S recall',points[r.NAMES[0]]['system_recall']!=points[r.NAMES[0]]['role_recall_system'])
# One component cannot produce an informative cluster CI.
one=counts.copy();one['dependency_component_id']='one'
paired,estimate,_=r.bootstrap(one)
check('single component CIs unavailable for every model/pair',all(row['ci_available'] is False and row['ci95_low'] is None and row['ci95_high'] is None and row['ci_unavailable_reason']=='fewer_than_two_dependency_components' for row in paired+estimate))
# Entire model with no system prediction has undefined precision, not zero.
never=counts.copy();mask=never.model==r.NAMES[0];never.loc[mask,'fn']=never.loc[mask,'tp']+never.loc[mask,'fn'];never.loc[mask,'tn']=never.loc[mask,'tn']+never.loc[mask,'fp'];never.loc[mask,['tp','fp']]=0
paired,estimate,_=r.bootstrap(never)
row=next(v for v in estimate if v['model']==r.NAMES[0] and v['metric']=='system_precision')
check('no-detection precision point and CI unavailable',row['value'] is None and not row['ci_available'] and row['defined_bootstrap_replicates']==0 and row['ci95_low'] is None)
check('undefined paired precision difference remains None',all(v['gain_percentage_points'] is None and v['defined_bootstrap_replicates']==0 for v in paired if v['baseline_model']==r.NAMES[0] and v['metric']=='system_precision'))
# No observations are a valid empty result, no fictitious cluster.
paired,estimate,frame=r.bootstrap(pd.DataFrame())
check('empty bootstrap returns empty tables',not paired and not estimate and frame.empty)
check('source unchanged during independent CPU bootstrap audit',start==hashlib.sha256(source.read_bytes()).hexdigest())
report=dict(source_sha256=start,checks=checks,checks_passed=len(checks),maximum_numeric_difference=maximum,
    paired_replicates=2000,paired_models=4,pairs=6,synthetic_only=True,real_test_data_read=False,c4_confirmation_data_read=False,gpu_used=False)
out=Path('/tmp/confirmation_review_natural_bootstrap_audit.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(out);print(json.dumps(report,sort_keys=True))
