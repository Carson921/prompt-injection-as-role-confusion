"""Source-only independent complementary-head mathematics; synthetic CPU only."""
import ast,copy,hashlib,itertools,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
SOURCE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis/systemness_complementary_head_scan.py')
PROPOSAL=SOURCE.parent/'outputs/local-gptoss120b/systemness-method-proposals/complementary-two-head-formal-proposal.json'
PROPOSAL_SHA='287f51d328da1bba3281313da750adb93af818bb37c9af26cad44d9159aab534'
DOMAINS=['C4-all','C4-tail','public-oasst','public-toxicchat0124'];ORIGINAL=[458,462,2045,460,459,2047,456,997]
FUNCTIONS=['binary_scores','unit_scores','aggregate','choose_pair','descriptive_bootstrap']
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,s):
    if not bool(v):raise RuntimeError(s)
def function_sources():
    text=SOURCE.read_text();tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    require({n.name for n in nodes}==set(FUNCTIONS),'Required pure source functions absent.')
    return text,tree,nodes,{n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest() for n in nodes}
text,tree,nodes,fnsha=function_sources();require(digest(PROPOSAL)==PROPOSAL_SHA,'Published prospective proposal changed.')
for n in tree.body:
    if isinstance(n,ast.Assign):
        for target in n.targets:
            if isinstance(target,ast.Name) and target.id in ['DOMAINS','ORIGINAL']:require(ast.literal_eval(n.value)==globals()[target.id],'Published domain/head order differs.')
# Extract only the five pure functions: do not import production/runtime/model code.
ns=dict(np=np,pd=pd,itertools=itertools,require=require,DOMAINS=DOMAINS,ORIGINAL=ORIGINAL)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
checks=[];hand=[];maximum_difference=0.
def check(s,v=True):require(v,s);checks.append(s)
def close(s,actual,expected,tol=2e-14):
    global maximum_difference
    d=abs(float(actual)-float(expected));maximum_difference=max(maximum_difference,d);check(s,d<=tol)
def independent(score,y,alpha=.05):
    y=np.asarray(y,bool);score=np.asarray(score);pos=int(y.sum());neg=int((~y).sum());blocks={}
    for value,label in zip(score,y):
        b=blocks.setdefault(float(value),[0,0]);b[0 if label else 1]+=1
    tp=0;fp=0;integral=0.
    for value,(p,n) in sorted(blocks.items(),reverse=True):
        left=fp/neg;right=(fp+n)/neg
        if n and left<alpha:
            width=max(0.,min(alpha,right)-left);slope=(p/pos)/(n/neg)
            integral+=(tp/pos)*width+.5*slope*width*width
        tp+=p;fp+=n
    auc=sum(float(a>b)+.5*float(a==b) for a in score[y] for b in score[~y])/(pos*neg)
    return dict(partial_auroc=integral/alpha,auroc=auc,balanced_nll=float(-.5*(np.log(np.clip(score[y].astype(np.float64),1e-15,1-1e-15)).mean()+np.log1p(-np.clip(score[~y].astype(np.float64),1e-15,1-1e-15)).mean())))
def case(name,p,y,want,auc):
    p=np.asarray(p,np.float32);y=np.asarray(y,bool);actual=ns['binary_scores'](p,y);own=independent(p,y)
    close(name+' hand partialAUROC',actual['partial_auroc'],want);close(name+' hand fullAUROC',actual['auroc'],auc)
    for k in actual:close(name+' independent '+k,actual[k],own[k])
    hand.append(dict(name=name,partial_auroc=want,full_auroc=auc,actual=actual))
case('Perfect FPR0 vertical', [.9,.8,.2,.1],[1,1,0,0],1.,1.)
case('Reverse ranking',[.1,.2,.8,.9],[1,1,0,0],0.,0.)
case('All ties chance integral is .025 not sklearn .5',[.5]*8,[1,1,0,0,0,0,0,0],.025,.5)
case('Partial initial vertical TPR .5',[.9,.1]+[.8]*20,[1,1]+[0]*20,.5,.5)
case('Boundary cuts through complete mixed tie',[.9,.8]+[.8]*2+[.1]*18,[1,1]+[0]*20,.625,.975)
case('Vertical at exact FPR .05 contributes zero area',[.9,.7,.8]+[.1]*19,[1,1,0]+[0]*19,.5,.975)
case('Several initial vertical groups',[.9,.8,.7,.1]+[.5]*20,[1]*4+[0]*20,.75,.75)
p=np.array([.9,.8,.8,.8,.1,.1,.1,.1],np.float32);y=np.array([1,1,0,0,1,0,0,0],bool);ref=ns['binary_scores'](p,y)
for seed in [3,19,41]:
    ix=np.random.RandomState(seed).permutation(len(p));got=ns['binary_scores'](p[ix],y[ix]);check('Complete ties ROC order invariance '+str(seed),got['partial_auroc']==ref['partial_auroc'] and got['auroc']==ref['auroc']);close('Balanced NLL reduction order '+str(seed),got['balanced_nll'],ref['balanced_nll'])
# Source->component->domain macro weights deliberately differ from token/source pooling.
items=[];scores=[]
def add(uid,domain,component,kind,repeat=1,dataset='C4'):
    for half in [0,1]:
        for label in [True,False]:
            for j in range(repeat):
                items.append(dict(source_uid=uid,domain=domain,component_id=component,dataset=dataset,role='system' if label else 'user',body_offset=half*512+j))
                scores.append(.5 if kind=='tied' else (.9 if label==(kind=='perfect') else .1))
add('c4a','C4','a','perfect',5);add('c4b','C4','b','reverse',1)
add('oa1','public','X','perfect',8,'oasst');add('oa2','public','X','reverse',1,'oasst');add('oa3','public','Y','perfect',1,'oasst')
add('tc1','public','X','tied',1,'toxicchat0124');add('tc2','public','Y','perfect',1,'toxicchat0124');add('tc3','public','Y','perfect',7,'toxicchat0124')
rows=pd.DataFrame(items);score=np.asarray(scores,np.float32);units,macro=ns['unit_scores'](rows,score)
for g,want in zip(DOMAINS,[.5,.5,.75,.5125]):close(g+' source/component/domain hand macro',macro[g]['partial_auroc'],want)
check('Both public domains preserve same dependency component IDs',set(units['public-oasst'])==set(units['public-toxicchat0124'])=={'X','Y'})
close('OA component X source-macro not token-pool',units['public-oasst']['X']['partial_auroc'],.5)
close('Public dataset weights are component-macro not conversation-macro',macro['public-oasst']['partial_auroc'],(.5+1.)/2)
# Every p/head aggregation is prescribed fixed-order F64 sum then one F32 cast.
rng=np.random.RandomState(713);prob=rng.uniform(size=(7,2304)).astype(np.float32)
for heads in [ORIGINAL,ORIGINAL+[0],ORIGINAL+[0,1]]:
    expected=np.zeros(7,np.float64)
    for h in heads:expected+=prob[:,h].astype(np.float64)
    expected=(expected/len(heads)).astype(np.float32);actual=ns['aggregate'](prob,heads)
    check('Fixed F64 sum /'+str(len(heads))+' then F32 bitexact',actual.dtype==np.float32 and np.array_equal(actual.view(np.uint32),expected.view(np.uint32)))
# Selection uses only synthetic marker columns. It evaluates exactly32 choose2.
# No actual head probabilities, rows, or production unit scores are used here.
marker=np.zeros((41,41),np.float32)
for h in range(8,41):marker[h,h]=1.
observed=[]
def metric_stub(_rows,s,screen_mode=False,no_admissible=False):
    active=np.flatnonzero(s);n=len(active);expected=np.float32(1/(8+n)) if n else 0.
    require(s.dtype==np.float32 and (not n or np.all(s[active]==expected)),'Incorrect marginal/pair /9 /10 F32 arithmetic.')
    observed.append(tuple(active.tolist()));gains=dict.fromkeys(DOMAINS,0.);fullgain=0.
    if n==1 and screen_mode:
        h=int(active[0])
        if h<=37:gains=dict.fromkeys(DOMAINS,.02)
        elif h==38:gains=dict(zip(DOMAINS,[0.,.01,0.,0.]))
        elif h==39:gains=dict(zip(DOMAINS,[0.,.02,0.,0.]))
        else:gains=dict.fromkeys(DOMAINS,.001)
    elif n==2 and not no_admissible:
        pair=tuple(active.tolist());gains=dict.fromkeys(DOMAINS,-.001);gains['C4-tail']=.006
        if pair==(8,9):gains['C4-tail']=.009;gains['C4-all']=-.002
        if pair==(8,10):gains['C4-tail']=.005-2e-12
        if pair in [(9,10),(9,11),(10,11),(10,12)]:
            gains=dict.fromkeys(DOMAINS,0. if pair==(9,10) else .001);gains['C4-tail']=.008;fullgain=.001 if pair in [(9,10),(9,11)] else .002
    metrics={d:dict(partial_auroc=.4+gains[d],auroc=.5+fullgain,balanced_nll=.7) for d in DOMAINS}
    return {d:{'toy-unit':metrics[d]} for d in DOMAINS},metrics
saved_unit=ns['unit_scores'];ns['unit_scores']=metric_stub
selection,marginal,pairs,_=ns['choose_pair'](pd.DataFrame(index=np.arange(41)),marker,original=list(range(8)),screen=32)
check('Exact32 screen then496 unordered distinct pairs',selection['screened_count']==32 and selection['pairs_evaluated']==len(pairs)==496 and all(len(set(x['head_ids']))==2 and x['head_ids']==sorted(x['head_ids']) for x in pairs))
check('Equal screen gains resolve by smaller headID',[*selection['screened_head_ids']]==list(range(8,40)))
check('Pair tail/worst/fullAUROC/lex hierarchy gives10,11',selection['selected_pair']['head_ids']==[10,11] and selection['selected_head_ids']==list(range(8))+[10,11])
by_pair={tuple(p['head_ids']):p for p in pairs};check('Other-domain loss and below-tail-floor reject despite ranks',not by_pair[(8,9)]['admissible'] and not by_pair[(8,10)]['admissible'])
check('Exactly -.001 other-domain losses are allowed at tolerance',by_pair[(8,11)]['admissible'])
ns['unit_scores']=lambda r,s:metric_stub(r,s,screen_mode=True)
edge,_,_,_=ns['choose_pair'](pd.DataFrame(index=np.arange(41)),marker,original=list(range(8)),screen=32)
check('Worst-domain priority then tail at screen edge excludes38',edge['screened_head_ids']==list(range(8,38))+[39,40])
ns['unit_scores']=lambda r,s:metric_stub(r,s,no_admissible=True)
stop,_,_,_=ns['choose_pair'](pd.DataFrame(index=np.arange(41)),marker,original=list(range(8)),screen=32)
check('No admissible pair stops without fallback or downstream heads',stop['selected_pair'] is None and stop['selected_head_ids'] is None and stop['stop_without_downstream_extraction'] and stop['pairs_evaluated']==496)
ns['unit_scores']=saved_unit
# Shared paired draws are verified independently for all/tail and OA/TC.
chosen=copy.deepcopy(units)
for domain in DOMAINS:
    for unit,v in chosen[domain].items():
        for key in v:v[key]+=(-.01 if v[key]>.5 else .02)
boot=ns['descriptive_bootstrap'](units,chosen);random=np.random.RandomState(20261021)
for domains in [DOMAINS[:2],DOMAINS[2:]]:
    ids=sorted(units[domains[0]]);draw=random.randint(0,len(ids),(2000,len(ids)))
    for domain in domains:
        for key in ['partial_auroc','auroc','balanced_nll']:
            delta=np.array([chosen[domain][u][key]-units[domain][u][key] for u in ids]);expected=np.quantile(delta[draw].mean(1),[.025,.975]);actual=boot['intervals'][domain][key]
            close(domain+'/'+key+' paired bootstrap mean',actual['mean_gain'],delta.mean());check(domain+'/'+key+' exact shared-draw percentiles',np.array_equal(actual['percentile95'],expected))
check('Bootstrap reports postselection optimism, no independent significance',boot['post_selection_optimistic'] and not boot['independent_significance'] and not boot['token_bootstrap'])
_,_,_,endsha=function_sources();check('Five audited math functions unchanged during annex',endsha==fnsha)
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-complementary-math-',dir='/tmp'))
report=dict(stage='independent_synthetic_complementary_head_math_audit',audit_passed=True,source_path=str(SOURCE),source_sha256_observed=digest(SOURCE),audited_function_sha256=fnsha,proposal_sha256=PROPOSAL_SHA,
    annex_source_sha256=digest(__file__),checks=checks,checks_passed=len(checks),hand_cases=hand,maximum_numeric_difference=maximum_difference,
    production_module_imported=False,actual_head_probabilities_or_fit_rows_read=False,validation_test_or_source_text_read=False,LM_or_GPU_used=False,formal_outputs_written=False,
    scope='Five extracted production pure functions only. Backend/source file may gain unrelated implementation later; preserve these function hashes or rerun annex.')
p=folder/'report.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(p),report_sha256=digest(p),**report)))
