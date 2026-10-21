"""Read-only CPU audit of actual head-scan development results; /tmp output only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
sys.path.insert(0,str(HERE))
import systemness_head_scan as s
import systemness_data as d
import systemness_long_data as ld
import systemness_long_probes as lp
import systemness_probes as probes
O=s.OUT;C=s.CACHE
DEST=Path('/tmp/systemness-head-results-audit.json')
SOURCE='6095e77d8ecf9cc22fb51169bdd126ec1f69efe9d3b59575b9e4b9094ba29ecd'
read=lambda p:json.loads(Path(p).read_text())
def check_file(path,expected):
 assert Path(path).is_file(),path
 value=d.digest(path);assert value==expected,(path,value,expected)
 return value

def save(report):
 temp=DEST.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(DEST)

def descriptor(inputs,rows):
 inputs=inputs.sort_values(['length','uid']).reset_index(drop=True)
 return dict(prompts=len(inputs),rows=len(rows),input_tokens=int(inputs.length.sum()),
  inputs_sha256=s.frame_digest(inputs,['uid','prompt','length']),
  rows_sha256=s.frame_digest(rows,['sample_ix','uid','token_ix','token_id','role','document_id']),
  sample_ix_sha256=s.digest_array(rows.sample_ix.to_numpy()))

def immutable_inputs():
 assert d.digest(s.__file__)==SOURCE
 plan=read(O/'plan.json');s.verify_plan(plan)
 assert not plan['tests_read'] and not plan['confirmation_inputs_read']
 assert plan['roles']==d.ROLES and plan['top_k']==8
 rows=pd.read_parquet(ld.CACHE/'rows.parquet',filters=[('split','=','fit')]).copy()
 inputs=pd.read_parquet(ld.CACHE/'inputs.parquet',filters=[('split','=','fit')]).copy()
 assert len(rows)==168960 and rows.document_id.nunique()==132 and set(rows.split)=={'fit'}
 rows['uid']='long-full:'+rows.prompt_ix.astype(str);inputs['uid']='long-full:'+inputs.prompt_ix.astype(str)
 rows['body_offset']=rows.token_ix-rows.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
 keys=read(ld.OUT/'document-keys.json');fit_ids={k['document_id'] for k in keys if k['split']=='fit'}
 val_ids={k['document_id'] for k in keys if k['split']=='validation'}
 assert set(rows.document_id)==fit_ids and not fit_ids & val_ids
 bundles={}
 for phase,mod,docs,count in [('fit',False,101,129280),('discovery',True,31,39680)]:
  r=rows[(rows.document_id%5==0)==mod].copy().reset_index(drop=True)
  i=inputs[inputs.uid.isin(r.uid)].copy()
  assert len(r)==count and r.document_id.nunique()==docs and r.groupby(['document_id','role']).size().eq(256).all()
  expected=d.uniform_indices(1024,256)
  assert all(np.array_equal(np.sort(g.body_offset),expected) for _,g in r.groupby(['document_id','role']))
  desc=descriptor(i,r);assert desc==plan['data'][phase]
  bundles[phase]=(i,r,desc)
 return plan,bundles

def scan():
 if not (O/'scan-complete.json').exists():print('SCAN_NOT_READY');return None
 started=time.time();plan,bundles=immutable_inputs();done=read(O/'scan-complete.json')
 hashes={str(O/'plan.json'):check_file(O/'plan.json',done['plan_sha256']),
  str(O/'head-selection.json'):check_file(O/'head-selection.json',done['head_selection_sha256']),
  str(O/'models/top8-lda.npz'):check_file(O/'models/top8-lda.npz',done['model_sha256']),
  str(O/'discovery-complete.json'):check_file(O/'discovery-complete.json',done['discovery_complete_sha256'])}
 moments={};phases={}
 for phase,nrows,ndocs in [('fit',129280,101),('discovery',39680,31)]:
  complete=read(O/f'{phase}-complete.json');assert complete['plan_sha256']==done['plan_sha256'] and not complete['tests_read']
  i,r,desc=bundles[phase];assert complete['data']==desc
  path=C/f'{phase}-moments.npz';hashes[str(path)]=check_file(path,complete['moments_sha256'])
  moments[phase],metadata=s.load_moments(path)
  expected=np.full((2304,5),ndocs*256,dtype=np.int64)
  assert np.array_equal(moments[phase].n,expected)
  assert metadata==dict(plan_sha256=done['plan_sha256'],phase=phase,next_batch=(len(i)+3)//4,
    rows_written=nrows,class_counts=[ndocs*256]*5,data=desc)
  phases[phase]=dict(rows=nrows,documents=ndocs,counts_per_head_per_role=[ndocs*256]*5,
   moments_sha256=complete['moments_sha256'],descriptor=desc)
 discovery=read(O/'discovery-complete.json');p_path=C/'discovery-system-probabilities.npy'
 hashes[str(p_path)]=check_file(p_path,discovery['probabilities_sha256'])
 hashes[str(O/'fit-lda-diagnostics.json')]=check_file(O/'fit-lda-diagnostics.json',discovery['fit_lda_diagnostics_sha256'])
 assert discovery['actual_probabilities_streamed'] and discovery['cpu_probability_max_difference']<1e-8
 p=np.load(p_path,mmap_mode='r');r=bundles['discovery'][1]
 assert p.shape==(39680,2304) and p.dtype==np.float32
 for start in range(0,len(p),1024):assert np.isfinite(p[start:start+1024]).all() and (p[start:start+1024]>=0).all() and (p[start:start+1024]<=1).all()
 ids=sorted(r.document_id.unique());all_auc=[];tail_auc=[];sklearn_difference=0.
 for docpos,doc in enumerate(ids):
  members=np.flatnonzero(r.document_id.to_numpy()==doc)
  for target,selected in [(all_auc,members),(tail_auc,members[r.iloc[members].body_offset.to_numpy()>=512])]:
   truth=r.iloc[selected].role.to_numpy()=='system';n=int(truth.sum());m=int((~truth).sum())
   assert (n,m) in [(256,1024),(128,512)]
   values=np.asarray(p[selected]);ranks=rankdata(values,axis=0,method='average')
   auc=(ranks[truth].sum(0)-n*(n+1)/2)/(n*m);target.append(auc)
   if docpos in [0,15,30]:
    for head in [0,458,462,997,2045,2047,2303]:
     sklearn_difference=max(sklearn_difference,abs(float(auc[head])-roc_auc_score(truth,values[:,head])))
  if (docpos+1)%8==0:print(json.dumps(dict(stage='independent-auroc',documents=docpos+1,total=len(ids))),flush=True)
 all_auc=np.array(all_auc);tail_auc=np.array(tail_auc);all_mean=all_auc.mean(0);tail_mean=tail_auc.mean(0);scores=(all_mean+tail_mean)/2
 order=np.lexsort((np.arange(2304),-tail_mean,-scores));ranks=np.empty(2304,int);ranks[order]=np.arange(1,2305)
 csv=pd.read_csv(O/'discovery-head-metrics.csv').sort_values('head_id').reset_index(drop=True)
 assert len(csv)==2304 and np.array_equal(csv.head_id,np.arange(2304))
 assert np.array_equal(csv.layer,np.arange(2304)//64) and np.array_equal(csv['head'],np.arange(2304)%64)
 differences={}
 for field,value in [('discovery_doc_macro_auroc',all_mean),('discovery_tail_doc_macro_auroc',tail_mean),('selection_score',scores)]:
  differences[field]=float(np.max(np.abs(csv[field].to_numpy()-value)));assert differences[field]<1e-14
 assert np.array_equal(csv['rank'],ranks)
 with np.load(C/'discovery-doc-auroc.npz',allow_pickle=False) as a:
  assert np.array_equal(a['document_ids'],ids) and np.array_equal(a['all_auc'],all_auc) and np.array_equal(a['tail_auc'],tail_auc)
 selected=order[:8];selection=read(O/'head-selection.json')
 assert selected.tolist()==selection['selected_head_ids']==done['selected_head_ids']
 assert selection['selected_layer_head_pairs']==[[int(h//64),int(h%64)] for h in selected]
 assert selection['model_sha256']==done['model_sha256'] and selection['probabilities_sha256']==discovery['probabilities_sha256']
 expected_scores=[dict(head_id=int(h),all_auroc=float(all_mean[h]),tail_auroc=float(tail_mean[h]),score=float(scores[h])) for h in selected]
 assert expected_scores==selection['selected_discovery_scores']
 combined={name:moments['fit'].arrays()[name][selected]+moments['discovery'].arrays()[name][selected] for name in ['n','sum','xx']}
 assert np.array_equal(combined['n'],np.full((8,5),132*256))
 fitted=s.HeadMoments(heads=8,arrays=combined);w,b,diag=fitted.fit(return_diagnostics=True)
 with np.load(O/'models/top8-lda.npz',allow_pickle=False) as a:
  modelmeta=json.loads(str(a['metadata']));mw=a['weight'];mb=a['bias']
  wdiff=float(np.abs(mw-w).max());bdiff=float(np.abs(mb-b).max())
  assert np.allclose(mw,w,atol=1e-10,rtol=1e-12) and np.allclose(mb,b,atol=1e-10,rtol=1e-12)
  assert np.array_equal(a['selected_head_ids'],selected)
  assert modelmeta['refit_lda_diagnostics']==diag==selection['refit_lda_diagnostics']
  assert modelmeta['fit_documents']==132 and modelmeta['discovery_documents']==31 and modelmeta['head_selection_fit_only']
  assert modelmeta['source_sha256']==SOURCE and modelmeta['plan_sha256']==done['plan_sha256']
 # A separate centered-within-class formula checks the refit covariance and priors.
 n,total,xx=combined['n'],combined['sum'],combined['xx'];mean=total/n[:,:,None]
 within=(xx-np.einsum('hkd,hke,hk->hde',total,total,1/n))/(n.sum(1)-5)[:,None,None]
 within=(within+within.transpose(0,2,1))/2
 grand=total.sum(1);count=n.sum(1)
 total_cov=(xx-np.einsum('hd,he,h->hde',grand,grand,1/count))/(count-1)[:,None,None]
 ridge=np.maximum.reduce([np.trace(within,axis1=1,axis2=2)/64,1e-6*np.trace(total_cov,axis1=1,axis2=2)/64,np.full(8,1e-12)])
 cov=.9*within+.1*ridge[:,None,None]*np.eye(64)
 rw=np.linalg.solve(cov,mean.transpose(0,2,1)).transpose(0,2,1)
 rb=-.5*np.einsum('hkd,hkd->hk',mean,rw)+np.log(n/n.sum(1,keepdims=True))
 assert np.allclose(w,rw,atol=1e-10,rtol=1e-12) and np.allclose(b,rb,atol=1e-10,rtol=1e-12)
 for phase in ['fit','discovery']:
  hook=read(O/f'hook-validation-{phase}.json');assert hook['hidden_states_bitexact'] and hook['statistics_disabled'] and hook['layers']==list(range(36))
 report=dict(pass_audit=True,source_sha256=SOURCE,plan_sha256=done['plan_sha256'],scan_complete_sha256=d.digest(O/'scan-complete.json'),
  probability_shape=list(p.shape),probability_bytes=int(p.nbytes),phases=phases,artifact_sha256=hashes,
  csv_entire2304_recomputed=True,csv_max_differences=differences,document_auc_npz_bitexact=True,
  sklearn_spotcheck_max_difference=sklearn_difference,rank_tie_order_verified=True,
  selected_head_ids=selected.tolist(),selected_layer_head_pairs=selection['selected_layer_head_pairs'],selected_discovery_scores=expected_scores,
  weights_max_difference=wdiff,bias_max_difference=bdiff,refit_counts_per_head_per_role=[132*256]*5,
  independent_covariance_refit=True,refit_diagnostics=diag,model_sha256=done['model_sha256'],
  actual_streamed_probabilities=True,discovery_portable_actual_max_difference=discovery['cpu_probability_max_difference'],
  hook_checks_statistics_disabled_and_hidden_bitexact=True,seconds=time.time()-started,
  limitation='Ranking used31 fit-discovery documents among2304 heads, not held-out validation. Refit uses all132 fit documents. No test/current464 rows read; no model/GPU loaded.')
 return report

def validation(scan_report):
 if not (O/'validation-complete.json').exists():print('VALIDATION_NOT_READY');return None
 started=time.time();plan=read(O/'plan.json');s.verify_plan(plan)
 assert d.digest(s.__file__)==SOURCE
 assert scan_report['scan_complete_sha256']==d.digest(O/'scan-complete.json')
 for path,sha in scan_report['artifact_sha256'].items():check_file(path,sha)
 done=read(O/'validation-complete.json');metrics=read(O/'validation-metrics.json')
 assert done['scan_complete_sha256']==scan_report['scan_complete_sha256'] and done['plan_sha256']==d.digest(O/'plan.json')
 check_file(O/'models/top8-lda.npz',done['model_sha256']);check_file(C/'validation-probabilities.npz',done['probabilities_sha256']);check_file(O/'validation-metrics.json',done['metrics_sha256'])
 assert not done['tests_read'] and not metrics['tests_read']
 rows=pd.read_parquet(lp.BANK/'rows.parquet',filters=[('split','=','validation')]).reset_index(drop=True)
 rows['uid']=rows.source_set+':'+rows.prompt_ix.astype(str)
 assert len(rows)==173032 and rows.sample_ix.is_unique and set(rows.split)=={'validation'} and set(rows.profile)==set(lp.PROFILES)
 assert done['data']['rows']==len(rows) and done['data']['sample_ix_sha256']==s.digest_array(rows.sample_ix.to_numpy())
 assert done['data']['rows_sha256']==s.frame_digest(rows,['sample_ix','uid','token_ix','token_id','role','document_id'])
 with np.load(C/'validation-probabilities.npz',allow_pickle=False) as a:
  assert np.array_equal(a['sample_ix'],rows.sample_ix);p=a['p'].copy()
 assert p.shape==(173032,5) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1.,atol=2e-5,rtol=0)
 result=lp.validation_metrics(rows,p);assert result==metrics['validation']
 assert all(result['profiles'][profile]['system_false_positive_rate']<=.05 for profile in lp.PROFILES)
 assert metrics['head_selection_sha256']==d.digest(O/'head-selection.json') and metrics['model_sha256']==scan_report['model_sha256']
 assert metrics['portable_probability_max_difference']<1e-8
 hook=read(O/'hook-validation-validation.json');assert hook['hidden_states_bitexact'] and hook['statistics_disabled']
 assert hook['layers']==sorted({h//64 for h in scan_report['selected_head_ids']})
 dense_plan=read(ld.OUT/'search-plan.json');dense_record=lp.candidate_record('dense-mlp128',dense_plan)
 dense_path=lp.BANK/'dense-mlp128-validation.npz'
 with np.load(dense_path,allow_pickle=False) as a:
  assert np.array_equal(a['sample_ix'],rows.sample_ix);dense_p=a['p'].copy()
 dense=lp.validation_metrics(rows,dense_p);assert dense==dense_record['validation']
 old={**dense['profiles'],'long-full-tail':dense['long_full_tail']};new={**result['profiles'],'long-full-tail':result['long_full_tail']}
 fields=['system_precision','system_recall','system_f1','system_balanced_accuracy','accuracy','system_false_positive_rate']
 comparisons={}
 for profile in new:
  a=old[profile];b=new[profile]
  comparisons[profile]={key:dict(dense128=a[key],top8_lda=b[key],delta_percentage_points=100*(b[key]-a[key])) for key in fields}
  comparisons[profile]['cot_recall']=dict(dense128=a['recall']['cot'],top8_lda=b['recall']['cot'],delta_percentage_points=100*(b['recall']['cot']-a['recall']['cot']))
 # Independent confusion/accuracy from saved probability at the single frozen threshold.
 independently=[];y=probes.labels(rows);threshold=result['system_threshold']
 for profile in lp.PROFILES:
  ix=np.flatnonzero(rows.profile.to_numpy()==profile);truth=y[ix]==0;detected=p[ix,0]>=threshold
  tp=int((truth&detected).sum());fp=int((~truth&detected).sum());fn=int((truth&~detected).sum());tn=int((~truth&~detected).sum())
  actual=dict(system_precision=tp/max(tp+fp,1),system_recall=tp/max(tp+fn,1),system_f1=2*tp/max(2*tp+fp+fn,1),system_balanced_accuracy=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),accuracy=float((p[ix].argmax(1)==y[ix]).mean()))
  assert all(np.isclose(v,result['profiles'][profile][k],atol=1e-14,rtol=0) for k,v in actual.items())
  independently.append(dict(profile=profile,tokens=len(ix),tp=tp,fp=fp,fn=fn,tn=tn,**actual))
 return dict(pass_audit=True,validation_complete_sha256=d.digest(O/'validation-complete.json'),model_sha256=done['model_sha256'],
  probabilities_sha256=done['probabilities_sha256'],validation_metrics_sha256=done['metrics_sha256'],sample_ix_sha256=s.digest_array(rows.sample_ix.to_numpy()),
  rows=len(rows),validation=result,dense128_validation=dense,comparisons=comparisons,
  mean5_f1_delta_percentage_points=100*(result['mean_system_f1']-dense['mean_system_f1']),
  independent_confusion=independently,threshold_and_all_metrics_recomputed=True,per_profile_fpr_at_most_05=True,
  actual_portable_evidence=dict(portable_probability_max_difference=metrics['portable_probability_max_difference'],
   discovery_actual_max_difference=scan_report['discovery_portable_actual_max_difference'],
   note='The production completion records compare portable NumPy against actual streamed GPU probabilities on captured rows; no raw head vectors persist, so CPU audit verifies bound evidence but does not replay model inference.'),
  dense128_model_sha256=dense_record['model_sha256'],dense128_probabilities_sha256=d.digest(dense_path),seconds=time.time()-started,
  limitation='Both are already-used development validation. Top8 LDA fit only132 long-full documents; dense128 fit578090 mixed rows. Classifier, train-content mixture, feature type/dimension and execution batches differ, so this is not an isolated feature causal ablation. No primary changes or new confirmation conclusions; no current464/old-test rows read.')

parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['scan','validation','all'],required=True);args=parser.parse_args()
base=read(DEST) if DEST.exists() else dict(audit_runner=str(Path(__file__)),audit_runner_sha256=d.digest(__file__),cpu_only=True,tests_read=False,current_confirmation_read=False)
assert base['audit_runner_sha256']==d.digest(__file__),'Audit runner changed since prior stage.'
if args.stage in ['scan','all']:
 if 'scan' not in base:
  value=scan()
  if value is not None:base['scan']=value;save(base);print(json.dumps(dict(scan_pass=True,path=str(DEST),selected=value['selected_head_ids'],scores=value['selected_discovery_scores']),indent=2),flush=True)
 else:print('SCAN_ALREADY_AUDITED_ONCE')
if args.stage in ['validation','all']:
 assert 'scan' in base,'Run --stage scan first.'
 value=validation(base['scan'])
 if value is not None:base['validation']=value;save(base);print(json.dumps(dict(validation_pass=True,path=str(DEST),mean5_delta_pp=value['mean5_f1_delta_percentage_points'],validation=value['validation'],comparisons=value['comparisons']),indent=2),flush=True)
