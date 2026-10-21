"""Fixed16-source DEVELOPMENT tomato gallery for prospective640 readouts; no fitting, LM or new test.

Default help. --preflight is synthetic CPU/tmp. --render is forbidden until root
pins both approved independent auditor and completed actual-audit SHAs below.
"""
import argparse,json,os,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import systemness_binary_development_plots as old
HERE=Path(__file__).resolve().parent;BASE=HERE/'outputs/local-gptoss120b'
TRAIN=BASE/'systemness-complementary-head-readout';OUT=BASE/'systemness-complementary-head-readout-development-plots'
PLAN=old.PLAN;PLAN_SHA=old.PLAN_SHA
HELPER_SHA='54798b2477de55e59b7f989ee210358057833a103822f817bbc7b6602fbba350'
TRAINER_SHA=None
PROTOCOL_SHA=None
DESIGN=BASE/'systemness-complementary-head-readout-preparation/root-prospective-design.json';DESIGN_SHA='f3dae68aae0003daf7382c925978313f984911cf428401d1bdd1eaa3b161c314'
APPROVED_AUDITOR_SHA=None;APPROVED_ACTUAL_AUDIT_SHA=None
MODELS=['oldhead','priorBinary','priorLongMix','extra2-new','duplicate-control'];RECORDS=['oldhead-max33','priorBinary-max33','priorLongMix-max33','new-max33','matched-max33']
ROLES=list(old.ROLES);COLORS=list(old.COLORS);CACHE=BASE/'systemness-expansion/cache'
PATHS={'oldhead':(CACHE/'natural-fit-probes','seed-20261021'),'priorBinary':(CACHE/'binary-residual','new'),'priorLongMix':(CACHE/'long-mix-residual','new'),'extra2-new':(CACHE/'complementary-head-readout','new'),'duplicate-control':(CACHE/'complementary-head-readout','matched')}
require,read,digest,array_sha,probability,counts=old.require,old.read,old.digest,old.array_sha,old.probability,old.counts


def gate(audit_path):
    require(all(v is not None for v in [TRAINER_SHA,PROTOCOL_SHA,APPROVED_AUDITOR_SHA,APPROVED_ACTUAL_AUDIT_SHA]),'Root must pin auditor and completed actual audit before rendering.')
    names={'systemness_complementary_head_readout.py','long_data_complementary_head_readout_results_audit.py'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        require(not any(Path(x.decode(errors='replace')).name in names for x in args if x),'Trainer/auditor live; no partial gallery reads.')
    require((TRAIN/'complete.json').is_file() and read(TRAIN/'status.json')['stage']=='validation_complete','Trainer not terminal complete.')
    require(digest(audit_path)==APPROVED_ACTUAL_AUDIT_SHA,'Actual development audit SHA changed.');a=read(audit_path)
    require(a['stage']=='actual_completed_complementary_head_readout_development_audit' and a['audit_passed'] is True and a['test_data_or_results_read'] is False and a['auditor_source_sha256']==APPROVED_AUDITOR_SHA,'Passing approved actual development audit required.')
    for path,key in [(TRAIN/'complete.json','trainer_complete_sha256'),(TRAIN/'protocol.json','trainer_protocol_sha256'),(TRAIN/'records.json','trainer_records_sha256'),(HERE/'systemness_complementary_head_readout.py','trainer_source_sha256')]:require(digest(path)==a[key],'Audited source/completion/thresholds changed.')
    require(a['trainer_source_sha256']==TRAINER_SHA and a['trainer_protocol_sha256']==PROTOCOL_SHA,'This gallery is bound to the frozen640 complementary-head trial.')
    require(digest(PLAN)==PLAN_SHA and digest(DESIGN)==DESIGN_SHA and digest(old.__file__)==HELPER_SHA,'Fixed source plan/design/helpers changed.')
    return a,read(PLAN)


def draw(plan,rows,values,thresholds,out):
    """Explicit pure renderer; old source globals remain untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    require(len(plan['entries'])==16 and len({e['slug'] for e in plan['entries']})==16 and set(values)==set(MODELS)==set(thresholds),'Fixed16/five-column coverage required.')
    groups=[]
    for e in plan['entries']:
        g=rows[e['bank']];g=g[g.prompt_ix.eq(e['prompt_ix'])].sort_values('token_ix')
        require(len(g)==e['observations'] and array_sha(g.sample_ix.to_numpy())==e['sample_ix_sha256'],'Fixed source membership/order changed.')
        if 'source_uid' in e:require(g.source_uid.eq(e['source_uid']).all(),'Fixed public source changed.')
        truth=np.asarray([ROLES.index(r) for r in g.role]);require(e['bank']!='public' or set(g.role)<={'system','user','assistant'},'Public CoT/tool truth invented.');groups.append((e,g,truth))
    out=Path(out);require(not (out/'complete.json').exists(),'Completed gallery cannot be overwritten.');out.mkdir(parents=True,exist_ok=True);table=[];files=[]
    for e,g,truth in groups:
        fig,axes=plt.subplots(5,5,figsize=(24,8),sharex=True,sharey=True);x=g.token_ix.to_numpy();ix=g.index.to_numpy()
        for col,model in enumerate(MODELS):
            p=values[model][e['bank']][ix];result=counts(p,truth,thresholds[model]);table.append(dict(example=e['slug'],cohort=e['cohort'],model=model,threshold=thresholds[model],**result))
            for role,label in enumerate(['Systemness','Userness','CoTness','Assistantness','Toolness']):
                ax=axes[role,col]
                for r,color in enumerate(COLORS):
                    mask=truth==r;ax.scatter(x[mask],p[mask,role],s=2.5,color=color,alpha=.7,rasterized=True)
                for boundary in g.groupby('message_ix').token_ix.min():ax.axvline(boundary,lw=.5,color='#d1d5db',zorder=0)
                if role==0:
                    ax.axhline(thresholds[model],color='#334155',ls='--',lw=.65)
                    if model=='oldhead':ax.axhline(.620969696969697,color='#64748b',ls=':',lw=.6)
                ax.set_ylim(-.03,1.03);ax.set_yticks([0,1],['0%','100%']);ax.spines[['top','right']].set_visible(False)
                if col==0:ax.set_ylabel(label,fontsize=9)
            f1='N/A' if result['system_f1'] is None else f"{result['system_f1']:.1%}"
            axes[0,col].set_title(f"{model} | five-class {result['five_accuracy']:.1%}\nsystem F1 {f1} | global tau {thresholds[model]:.3f}",fontsize=9);axes[-1,col].set_xlabel('Token position in native input',fontsize=9)
        fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markersize=4,label=r) for r,c in zip(ROLES,COLORS)],loc='upper center',ncol=5,frameon=False,fontsize=9)
        fig.suptitle('DEVELOPMENT ONLY | '+e['slug']+' | '+e.get('source_uid','legacy prompt '+str(e['prompt_ix'])),y=.943,fontsize=10)
        fig.text(.5,.008,'Unchanged fixed16 ranks; unsmoothed content-token probabilities, gaps never filled. Public S/U/A has no CoT/tool truth. Dashed: max33; dotted oldhead: original tau.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.03,1,.90])
        for ext in ['png','pdf']:
            p=out/(e['slug']+'.'+ext);fig.savefig(p,dpi=140);files.append(p)
        plt.close(fig)
    require(len(table)==80 and len(files)==32,'Fixed16/five-model gallery incomplete.');pd.DataFrame(table).to_csv(out/'example-metrics.csv',index=False)
    (out/'gallery-index.md').write_text('\n'.join(['Fixed16-source DEVELOPMENT gallery: five readouts','Reused development only; not independent confirmation. Same16 sources, no replacements.','']+[f"- [{e['slug']}]({e['slug']}.png) ([PDF]({e['slug']}.pdf))" for e in plan['entries']])+'\n');return table,files+[out/'example-metrics.csv',out/'gallery-index.md']


def render(audit_path):
    audit,plan=gate(audit_path);require(not (OUT/'complete.json').exists(),'Gallery already complete.');rows={}
    for bank,path,count in [('old',Path('/dev/shm/role-analysis-top8-heads/rows.parquet'),173032),('public',Path('/dev/shm/role-analysis-natural-dev-heads/rows.parquet'),16713)]:
        require(digest(path)==plan['rows_sha256'][bank] and audit['provenance_sha256'][str(path.resolve())]==plan['rows_sha256'][bank],'Audited fixed row bank changed.');frame=pd.read_parquet(path);rows[bank]=frame[frame.split.eq('validation')].reset_index(drop=True);require(len(rows[bank])==count,'Developer cohort changed.')
    records={r['record']:r for r in read(TRAIN/'records.json')};thresholds={m:records[r]['validation']['system_threshold'] for m,r in zip(MODELS,RECORDS)}
    require(all(np.isfinite(v) and 0<=v<=1.000001 for v in thresholds.values()),'Invalid frozen global threshold.')
    values={m:{b:probability(audit,folder/f'{prefix}-{b}-validation.npz',rows[b]) for b in rows} for m,(folder,prefix) in PATHS.items()}
    table,files=draw(plan,rows,values,thresholds,OUT);gate(audit_path)
    done=dict(stage='completed_fixed_complementary_head_development_gallery',examples=16,readouts=MODELS,png=16,pdf=16,metrics=80,source_ranks_changed=False,independent_test=False,LM_run=False,probabilities_recomputed=False,
        audit_sha256=digest(audit_path),auditor_source_sha256=APPROVED_AUDITOR_SHA,trainer_complete_sha256=audit['trainer_complete_sha256'],plot_source_sha256=digest(__file__),helper_sha256=HELPER_SHA,gallery_plan_sha256=PLAN_SHA,
        inputs={str((folder/f'{prefix}-{b}-validation.npz').resolve()):digest(folder/f'{prefix}-{b}-validation.npz') for folder,prefix in PATHS.values() for b in rows},artifacts={p.name:digest(p) for p in files})
    (OUT/'complete.json').write_text(json.dumps(done,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(out=str(OUT),png=16,pdf=16,metrics=80,complete_sha256=digest(OUT/'complete.json'))))


def preflight():
    from unittest.mock import patch
    import sys,py_compile
    folder=Path(tempfile.mkdtemp(prefix='complementary-head-gallery-preflight-',dir='/tmp'));source=digest(__file__);checks=[];original=(old.OUT,old.TRAIN,list(old.MODELS))
    def check(s,v=True):require(v,s);checks.append(s)
    def forbidden(*a,**k):raise LookupError('Formal metadata/p access forbidden in mock preflight.')
    with patch.object(sys.modules[__name__],'APPROVED_AUDITOR_SHA',None),patch.object(sys.modules[__name__],'APPROVED_ACTUAL_AUDIT_SHA',None),patch.object(sys.modules[__name__],'read',side_effect=forbidden),patch.object(sys.modules[__name__],'digest',side_effect=forbidden):
        try:gate(folder/'missing-audit.json')
        except RuntimeError:check('Unpinned actual gate refuses before metadata/probability reads')
        else:raise RuntimeError('Unpinned actual accepted')
    require(digest(PLAN)==PLAN_SHA and digest(old.__file__)==HELPER_SHA,'Fixed plan/helper changed.');fixed=read(PLAN);entries=[];data={'old':[],'public':[]}
    for e in fixed['entries']:
        bank=e['bank'];roles=['system','user','assistant'] if bank=='public' else ['system','user','cot','assistant'];items=[]
        for i,r in enumerate(roles):items.append(dict(prompt_ix=e['prompt_ix'],token_ix=10+i*13,sample_ix=len(data[bank])+i,role=r,message_ix=i,source_uid=e.get('source_uid','synthetic')))
        g=pd.DataFrame(items);data[bank]+=items;entries.append(dict(e,observations=len(g),sample_ix_sha256=array_sha(g.sample_ix.to_numpy())))
    rows={b:pd.DataFrame(v) for b,v in data.items()};values={}
    for m in MODELS:
        values[m]={}
        for b,f in rows.items():
            p=np.full((len(f),5),.025,np.float32);p[np.arange(len(f)),[ROLES.index(r) for r in f.role]]=.9;values[m][b]=p
    table,files=draw(dict(entries=entries),rows,values,{m:.5 for m in MODELS},folder/'mock-gallery')
    check('Same16 fixed source IDs retained',[(e['slug'],e['prompt_ix']) for e in entries]==[(e['slug'],e['prompt_ix']) for e in fixed['entries']])
    check('16PNG/PDF and80 examples complete',len(table)==80 and len([p for p in files if p.suffix=='.png'])==16 and len([p for p in files if p.suffix=='.pdf'])==16)
    check('Public truth zeroCoT support despite5probability axes',all(v['cot_support']==0 for v in table if v['cohort']!='legacy-natural-CoT') and all(v['cot_support']>0 for v in table if v['cohort']=='legacy-natural-CoT'))
    check('Original helper globals unchanged',(old.OUT,old.TRAIN,old.MODELS)==original);py_compile.compile(__file__,cfile=str(folder/'gallery.pyc'),doraise=True);require(digest(__file__)==source,'Mock changed source.')
    report=dict(source_sha256=source,checks=checks,checks_passed=len(checks),png=16,pdf=16,metrics=80,fixture_directory=str(folder),actual_probabilities_or_rows_read=False,gpu_used=False,formal_outputs_written=False)
    p=folder/'report.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),**report)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group();m.add_argument('--render',action='store_true');m.add_argument('--preflight',action='store_true');p.add_argument('--audit',type=Path);a=p.parse_args()
    if a.preflight:preflight()
    elif a.render:require(a.audit is not None,'Provide independently completed actual audit.');render(a.audit)
    else:p.print_help()
