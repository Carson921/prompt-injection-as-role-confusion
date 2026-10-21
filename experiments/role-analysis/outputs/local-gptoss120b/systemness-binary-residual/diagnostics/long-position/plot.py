#!/usr/bin/env python3
"""Four-bin frozen-readout DEVELOPMENT diagnostic; no models/LM/inference/test."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/prompt-injection-as-role-confusion')
BASE=ROOT/'experiments/role-analysis/outputs/local-gptoss120b'
TRAIN=BASE/'systemness-binary-residual'
CACHE=BASE/'systemness-expansion/cache'
AUDIT=Path('/tmp/long_data_binary_residual_results_audit.json')
AUDIT_SHA='2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67'
ROWS=Path('/dev/shm/role-analysis-long-augmentation-heads/rows.parquet')
OUT=Path('/tmp/feature_review_binary_long_position')
SPEC=[('oldhead','oldhead-max33',CACHE/'natural-fit-probes/seed-20261021-c4-validation.npz'),
      ('naturalCE','naturalCE-max33',CACHE/'natural-fit-probes/new-c4-validation.npz'),
      ('ensemble','ensemble-max33',CACHE/'natural-fit-probes/ensemble-c4-validation.npz'),
      ('binary-new','new-max33',CACHE/'binary-residual/new-c4-validation.npz'),
      ('binary-matched','matched-max33',CACHE/'binary-residual/matched-c4-validation.npz')]
COLORS=['#64748b','#d97706','#2563eb','#059669','#a855f7']
BIN_LABELS=['0–255','256–511','512–767','768–1023']

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,why):
    if not bool(ok):raise RuntimeError(why)

require(digest(AUDIT)==AUDIT_SHA,'Approved actual audit changed.')
audit=read(AUDIT)
require(audit['stage']=='actual_completed_binary_residual_development_audit' and audit['audit_passed'] is True
        and audit['test_data_or_results_read'] is False,'Passing audited development certificate required.')
require(read(TRAIN/'status.json')['stage']=='validation_complete','Trainer not terminal.')
bindings={str(AUDIT):AUDIT_SHA,str(Path(__file__).resolve()):digest(__file__)}
def bind(path,want):
    require(digest(path)==want,'Audited input changed: '+str(path));bindings[str(Path(path).resolve())]=want
bind(TRAIN/'complete.json',audit['trainer_complete_sha256'])
bind(TRAIN/'protocol.json',audit['trainer_protocol_sha256'])
bind(TRAIN/'records.json',audit['trainer_records_sha256'])
bind(ROWS,audit['provenance_sha256'][str(ROWS.resolve())])
rows=pd.read_parquet(ROWS)
rows=rows[rows.split.eq('validation')].reset_index(drop=True)
require(len(rows)==61440 and rows.sample_ix.is_unique and rows.profile.eq('long-full').all()
        and rows.wrapper.eq('native').all() and rows.document_id.nunique()==96
        and rows.body_offset.between(0,1023).all(),'Expected native96-source sampled long validation rows.')
roles=['system','user','cot','assistant','tool']
require(set(rows.role)==set(roles),'Five roles required.')
per=rows.groupby(['document_id','role']).size()
require(len(per)==96*5 and per.eq(128).all(),'Exactly128 sampled observations per role per1024-token source required.')
bin_ix=(rows.body_offset.to_numpy()//256).astype(int)
require(np.array_equal(np.bincount(bin_ix,minlength=4),np.repeat(15360,4)),'Position bins must sum61440 with15360 each.')
for i in range(4):require(rows.iloc[np.flatnonzero(bin_ix==i)].document_id.nunique()==96,'Every bin must contain all96 sources.')
truth=rows.role.eq('system').to_numpy()
records={r['record']:r['validation'] for r in read(TRAIN/'records.json')}
table=[]
for name,record,path in SPEC:
    bind(path,audit['probability_bindings'][str(path.resolve())])
    with np.load(path,allow_pickle=False) as z:
        require(set(z.files)=={'p','sample_ix'} and np.array_equal(z['sample_ix'],rows.sample_ix.to_numpy()),'Probability sample identity/order changed.')
        p=z['p'].copy()
    require(p.dtype==np.float32 and p.shape==(61440,5) and np.isfinite(p).all() and (p>=0).all()
            and (p<=1).all() and np.allclose(p.sum(1),1,atol=2e-5,rtol=0),'Invalid audited p.')
    threshold=records[record]['system_threshold'];detect=p[:,0]>=float(threshold)
    for i in range(4):
        m=bin_ix==i;t=truth[m];d=detect[m]
        tp=int((t&d).sum());fp=int((~t&d).sum());fn=int((t&~d).sum());tn=int((~t&~d).sum())
        require(tp+fn==3072 and fp+tn==12288,'Bin formal role supports changed.')
        table.append(dict(model=name,frozen_record=record,system_threshold=threshold,body_offset_first=256*i,
                          body_offset_last=256*i+255,sources=96,sampled_observations=tp+fp+fn+tn,
                          system_support=tp+fn,non_system_support=fp+tn,tp=tp,fp=fp,fn=fn,tn=tn,
                          system_precision=tp/(tp+fp) if tp+fp else None,system_recall=tp/(tp+fn),
                          system_f1=2*tp/(2*tp+fp+fn),system_balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2))
require(len(table)==20,'Exactly20 frozen-readout/bin records required.')
for name,_,_ in SPEC:require(sum(v['sampled_observations'] for v in table if v['model']==name)==61440,'Per-model bin counts do not sum61440.')
OUT.mkdir(exist_ok=True)
csv_path=OUT/'long-position-metrics.csv'
with csv_path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12.8,8.0),sharex=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.23,top=.86,hspace=.38,wspace=.22)
metrics=[('system_precision','System precision'),('system_recall','System recall'),
         ('system_f1','System F1'),('system_balanced_accuracy','System balanced accuracy')]
for ax,(key,label) in zip(axes.flat,metrics):
    for color,(name,_,_) in zip(COLORS,SPEC):
        values=[x[key] for x in table if x['model']==name]
        ax.plot(range(4),values,marker='o',markersize=5,linewidth=1.35,color=color,label=name)
    ax.axhline(.90,color='#334155',ls='--',lw=.8)
    ax.set_title(label,loc='left');ax.set_xticks(range(4),BIN_LABELS)
    ax.set_xlabel('Observed original body-token offsets');ax.set_ylabel(label)
    lo=min(x[key] for x in table);ax.set_ylim(max(0.,min(.88,lo-.012)),1.005)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',alpha=.2)
fig.suptitle('Long-position diagnostic · DEVELOPMENT ONLY\n96 fixed 1024-token C4 sources · frozen max33 thresholds',fontsize=14)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncols=5,frameon=False,bbox_to_anchor=(.5,.09))
fig.supxlabel('Each role has only 128 sampled observations per source (32 per bin), not full-token evaluation.\nAll 96 sources retained; raw bin statistics, no smoothing or CI; reused development data, not a new test.',fontsize=9,y=.016)
artifacts={}
for suffix in ['png','pdf']:
    path=OUT/('long-position.'+suffix);fig.savefig(path,dpi=160);artifacts[path.name]=digest(path)
plt.close(fig);artifacts[csv_path.name]=digest(csv_path)
require(all(digest(path)==value for path,value in bindings.items()),'Input/source changed during rendering.')
publication=dict(stage='completed_fixed_long_position_development_diagnostic',audit_sha256=AUDIT_SHA,
    source_sha256=digest(__file__),source_path=str(Path(__file__).resolve()),provenance_sha256=bindings,
    models=[s[0] for s in SPEC],frozen_records=[s[1] for s in SPEC],sources=96,actual_original_body_tokens=1024,
    sampled_tokens_per_role_per_source=128,sampled_tokens_per_role_per_source_per_bin=32,
    observations_per_model=61440,bin_observations=[15360]*4,csv_rows=20,panels=4,PNG=1,PDF=1,
    binary_thresholds_retuned=False,models_recomputed=False,model_or_LM_loaded=False,GPU_used=False,
    new_test=False,independent_confirmation=False,smoothing=False,confidence_intervals=False,
    current_fixed16_gallery_modified=False,formal_outputs_modified=False,
    truth='Five formal native roles; system binary metrics against all four non-system roles.',
    limitation='These fixed1024-token source bodies are sampled at128 content positions per role. Bins are overlapping-source summaries of reused development data; no full-token/new-independent-test claim.',
    artifacts=artifacts)
dest=OUT/'publication.json';dest.write_text(json.dumps(publication,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(directory=str(OUT),script=str(Path(__file__).resolve()),script_sha256=digest(__file__),
                      publication=str(dest),publication_sha256=digest(dest),artifacts=artifacts,rows=len(table))))
