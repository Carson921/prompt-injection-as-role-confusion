"""Natural-fit publication planner. Default help; never trains or reads predictions.

--dry-run writes a reviewable plan under /tmp only, after terminal training and a
pinned independent actual audit. --apply requires the exact reviewed plan SHA;
it rechecks the same closed files, then copies only the fixed allowlist into
OUT/preparation. Existing differing destination bytes are never overwritten.
--preflight uses synthetic metadata/models under /tmp, never formal results.
"""
import argparse,csv,fcntl,hashlib,json,os,shutil,tempfile,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path('/root/prompt-injection-as-role-confusion')
HERE=ROOT/'experiments/role-analysis';BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-natural-fit-probes';DEST=OUT/'preparation'
SOURCE=HERE/'systemness_natural_fit_probes.py'
AUDITOR=Path('/tmp/long_data_natural_fit_results_audit.py')
AUDIT=Path('/tmp/long_data_natural_fit_results_audit.json')
SOURCE_SHA='7c0674adbce3ef4980a1f29077a6ca36396b93aa992c32ad848ff9e56be4aea3'
PROTOCOL_SHA='1edc202c1f8f2651e730a650590489c5a8f1f8da1b9da4ab730f0b949b59c104'
AUDITOR_SHA='0d755e9491ecf65f5477714140260e4ebc45319d413c3a64ee72a18bc426ad57'
MAX_FILE=50*2**20
ARTIFACTS=['protocol.json','search-plan.json','draw-manifest.json','controls.json','records.json','selection.json','validation-metrics.csv','source-counts.json','report.txt']
RECORDS=['new-oldmean','new-max33','matched-oldmean','matched-max33','oldhead-oldmean','oldhead-max33','ensemble-original','ensemble-max33']
GROUPS=['old-single','old-multi','old-natural','old-long-single','old-long-full','old-long-full-tail','new-C4-long-full','new-C4-long-full-tail','public-natural-all','public-natural-oasst','public-natural-toxicchat0124']
EVIDENCE=[
 ('auditor-source',AUDITOR,AUDITOR_SHA,'audits/actual-auditor.py'),
 ('auditor-44CPU',Path('/tmp/long-data-natural-fit-audit-preflight-7jmtxzgv/report.json'),'58289dd723eaaef61599576206eeac6b7a6146d23166f78ebe5d4b65b1ad722f','audits/auditor-44CPU.json'),
 ('auditor-15annex',Path('/tmp/confirmation-review-natural-fit-auditor-r6rwqolt/report.json'),'7f2c9a4c490411291fb1bdc736c99dc900787b9d86d09fd2e47124320d34c269','audits/auditor-15annex.json'),
 ('annex-source',Path('/tmp/confirmation_review_natural_fit_auditor_annex.py'),'d4b45c4f6cb88ce12315b7b6413e3b35c34432d4d0f90824b17a9ba15775457d','audits/auditor-annex.py'),
 ('trainer-28CPU',Path('/tmp/systemness-natural-fit-preflight-12ncid7a/report.json'),'bada48be6296a5c7e7a55e9e324dab2d325aa75d1e4a7180bf44a85b4bb8e8a7','audits/trainer-28CPU.json'),
 ('trainer-27annex',Path('/tmp/feature-review-natural-fit-source-audit-457yovdm/report.json'),'79efd533878c9c467cf557c81de3ee9233caf2ca0cd6a444406b123a04235114','audits/trainer-27annex.json'),
 ('trainer-annex-source',Path('/tmp/feature_review_natural_fit_source_audit.py'),'bb6f8a4edb598ad5fb53ce08befe72c01137d585260f1742e54494e2f7ab3985','audits/trainer-annex.py')]


def require(v,msg):
 if not v:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 require(not tmp.exists(),'Stale temporary file requires inspection: '+str(tmp))
 with tmp.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def processes():
 active=[]
 names={'systemness_natural_fit_probes.py','systemness_natural_dev_extract.py','long_data_natural_fit_results_audit.py'}
 for entry in Path('/proc').iterdir():
  if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
  try:args=(entry/'cmdline').read_bytes().split(b'\0')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if any(Path(v.decode(errors='replace')).name in names for v in args if v):active.append(int(entry.name))
 return active

def context(audit=AUDIT):
 return SimpleNamespace(out=OUT,dest=DEST,source=SOURCE,source_sha=SOURCE_SHA,protocol_sha=PROTOCOL_SHA,
    auditor=AUDITOR,auditor_sha=AUDITOR_SHA,audit=Path(audit),evidence=EVIDENCE,helper=Path(__file__).resolve())

def terminal(ctx):
 require(not processes(),'Training/extraction/actual auditor is still live; no release reads.')
 require((ctx.out/'complete.json').is_file() and (ctx.out/'status.json').is_file(),'Terminal completion/status missing.')
 require(read(ctx.out/'status.json')['stage']=='validation_complete','Training is not validation_complete.')
 require(ctx.audit.is_file(),'Independent actual audit is not present.')
 audit=read(ctx.audit)
 require(audit['stage']=='actual_completed_natural_fit_development_audit' and audit['audit_passed'] is True
    and audit['test_data_or_results_read'] is False and audit['gpu_used'] is False and audit['formal_outputs_written'] is False,
    'Independent actual audit is not a passing developer-only certificate.')
 return audit

def model_zip(path):
 expected={'mean.npy','scale.npy','metadata.npy','weight::encoder.0.weight.npy','weight::encoder.0.bias.npy','weight::head.weight.npy','weight::head.bias.npy'}
 with zipfile.ZipFile(path) as archive:require(set(archive.namelist())==expected,'Only portable MLP model NPZ is publishable; prediction/feature NPZ excluded.')

def eligible(path,model=False):
 path=Path(path)
 if path.is_symlink():return False,'symlink'
 if not path.is_file():return False,'missing'
 if path.stat().st_size>=MAX_FILE:return False,'at_least_50MiB'
 if path.suffix in {'.f16','.f32','.npy','.lock'} or path.name.startswith('.') or 'probabilities' in path.name or 'validation.npz' in path.name:return False,'raw_prediction_or_lock'
 if path.suffix=='.npz' and not model:return False,'NPZ_not_explicit_portable_model'
 if model:model_zip(path)
 return True,None

def build_plan(ctx):
 audit=terminal(ctx) # Never import a production verifier or scan feature/prediction directories.
 complete=read(ctx.out/'complete.json');selection=read(ctx.out/'selection.json');records=read(ctx.out/'records.json')
 require(digest(ctx.source)==ctx.source_sha==audit['trainer_source_sha256'] and digest(ctx.out/'protocol.json')==ctx.protocol_sha==audit['trainer_protocol_sha256'],
    'Pinned trainer/protocol differs from independent audit.')
 require(digest(ctx.auditor)==ctx.auditor_sha==audit['auditor_source_sha256'],'Pinned independent auditor source differs.')
 require(digest(ctx.out/'complete.json')==audit['trainer_complete_sha256'] and digest(ctx.out/'selection.json')==audit['trainer_selection_sha256']
    and digest(ctx.out/'records.json')==audit['trainer_records_sha256'],'Result completion/selection/records changed after actual audit.')
 require(complete['source_sha256']==ctx.source_sha and complete['tests_read'] is False and complete['primary_changed'] is False
    and complete['new_independent_confirmation_required'] is True and set(complete['artifact_sha256'])==set(ARTIFACTS)
    and set(complete['candidate_sha256'])=={'new','matched'},'Trainer completion boundary/coverage changed.')
 require(selection['candidate']=='new-max33' and isinstance(selection['promoted'],bool) and selection['primary_changed'] is False
    and selection['tests_read'] is False and selection['no_fallback'] is True and selection['new_independent_confirmation_required'] is True,
    'Selection boundary changed; publication never selects a readout.')
 require(len(records)==8 and {r['record'] for r in records}==set(RECORDS) and audit['csv_primary_group_rows']==88 and audit['csv_descriptive_pool_rows']==8,'88+8 metric coverage changed.')
 with (ctx.out/'validation-metrics.csv').open(newline='') as f:csvrows=list(csv.DictReader(f))
 require(len(csvrows)==96 and len({(r['record'],r['group']) for r in csvrows})==96
    and {(r['record'],r['group']) for r in csvrows}=={(r,g) for r in RECORDS for g in GROUPS+['pooled_descriptive_only']},'CSV row mapping differs.')
 files=[];excluded=[]
 def add(path,destination,kind,expected=None,model=False):
  path=Path(path);ok,reason=eligible(path,model)
  if not ok:excluded.append(dict(path=str(path),reason=reason,required=True));return
  value=digest(path)
  if expected is not None:require(value==expected,'Required publication artifact SHA differs: '+str(path))
  declared=audit['provenance_sha256'].get(str(path.resolve()))
  if declared is not None:require(value==declared,'Artifact changed since independent audit: '+str(path))
  require(not Path(destination).is_absolute() and '..' not in Path(destination).parts,'Unsafe publication path.')
  files.append(dict(source=str(path.resolve()),destination=destination,kind=kind,bytes=path.stat().st_size,sha256=value))
 for n in ARTIFACTS:add(ctx.out/n,'results/'+n,'frozen-result-metadata',complete['artifact_sha256'][n])
 add(ctx.out/'complete.json','results/complete.json','terminal-completion',audit['trainer_complete_sha256'])
 add(ctx.out/'status.json','results/status.json','terminal-status')
 for arm in ['new','matched']:
  path=ctx.out/'candidates'/f'{arm}.json';candidate=read(path)
  require(digest(path)==complete['candidate_sha256'][arm]==audit['trainer_model_record_sha256'][arm],'Candidate record changed after audit.')
  require(candidate['model_sha256']==audit['model_sha256'][arm],'Candidate portable model differs from independently checked model.')
  add(path,f'candidates/{arm}.json','candidate-budget-and-GPU-CPU-parity',complete['candidate_sha256'][arm])
  add(ctx.out/'training'/f'{arm}.json',f'training/{arm}.json','full-epoch-history-and-best-freeze',candidate['training_sha256'])
  add(ctx.out/'models'/f'{arm}.npz',f'models/{arm}.npz','portable-MLP-model',candidate['model_sha256'],True)
 add(ctx.source,'source/'+ctx.source.name,'frozen-trainer',ctx.source_sha)
 add(ctx.audit,'audits/actual-development-audit.json','independent-actual-audit',digest(ctx.audit))
 for kind,path,want,target in ctx.evidence:add(path,target,kind,want)
 add(ctx.helper,'source/publication-helper.py','publication-helper',digest(ctx.helper))
 require(len({f['destination'] for f in files})==len(files),'Duplicate release destination.')
 plan=dict(version=1,stage='reviewable_natural_fit_publication_plan',destination=str(ctx.dest.resolve()),
    files=sorted(files,key=lambda f:f['destination']),excluded=excluded,required_files_complete=not excluded,total_bytes=sum(f['bytes'] for f in files),
    pins=dict(trainer_source=ctx.source_sha,protocol=ctx.protocol_sha,auditor_source=ctx.auditor_sha,actual_audit=digest(ctx.audit),complete=audit['trainer_complete_sha256']),
    copied_models=['new','matched'],inherited_models='Referenced by existing frozen model paths/hashes in controls/audit; not duplicated.',
    payload_policy='Fixed allowlist only. Never walk/cache/copy raw banks, rows/input text caches, prediction NPZ, portable feature samples or lock files. Required>=50MiB files are excluded and apply is blocked.',
    raw_features_or_probabilities_rehashed=False,raw_feature_probability_SHA_provenance='Relies on pinned passing actual audit; only publishable closed files are rehashed here.',
    root_interpretation_required=[
      'Explain promoted flag and every absolute90/strict matched-oldhead-ensemble/public/C4/mandatory savedGPU rejection reason; never fallback to another record.',
      'Interpret all8 developer records/88 groups+8 pooled descriptions; distinguish min33 improvement from independent goal completion.',
      'Report same-role/update matching limits, shared seed/scaler, public source/family/component counts, and repeated developer validation.',
      'Keep legacy old-natural CoT support separate from public SUA-only truth; do not claim assistant replies were generated by120B or under assigned wording.',
      'Report best versus actual steps and GPU/CPU drift/argmax/original-threshold changes; below-tolerance drift can still change decisions.',
      'Current primary stays unchanged. Any qualifying candidate still needs fresh disjoint C4+natural confirmation and fixed tomato plots.'],
    no_primary_change=True,no_model_or_threshold_selection=True,no_git_or_historical_MD_changes=True)
 plan['content_sha256']=canonical(plan);return plan

def apply_plan(ctx,plan,reviewed_sha,path):
 require(digest(path)==reviewed_sha,'Reviewed plan file SHA differs.')
 require(plan==build_plan(ctx),'Release inputs changed since reviewed dry-run.')
 require(plan['required_files_complete'],'A required file was missing/excluded; publication is incomplete.')
 ctx.dest.mkdir(parents=True,exist_ok=True)
 with (ctx.dest/'.publication.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(plan==build_plan(ctx),'Release inputs changed before copying.')
  pending=sum(e['bytes'] for e in plan['files'] if not (ctx.dest/e['destination']).exists())
  require(shutil.disk_usage(ctx.dest).free>=pending+64*2**20,'Insufficient free disk for bounded publication.')
  for entry in plan['files']:
   source=Path(entry['source']);target=ctx.dest/entry['destination'];require(digest(source)==entry['sha256'],'Source changed before copying.')
   if target.exists():require(not target.is_symlink() and digest(target)==entry['sha256'],'Existing destination differs; refusing overwrite.');continue
   target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.tmp');require(not tmp.exists(),'Stale copy temp requires inspection.')
   with source.open('rb') as inp,tmp.open('xb') as out:shutil.copyfileobj(inp,out);out.flush();os.fsync(out.fileno())
   require(digest(tmp)==entry['sha256'],'Copy SHA mismatch.');tmp.replace(target)
  require(plan==build_plan(ctx),'Source closure changed while copying.')
  marker=ctx.dest/'publication-manifest.json'
  if marker.exists():require(read(marker)==plan,'Existing publication manifest differs.')
  else:atomic_json(marker,plan)
 return dict(manifest=str(marker),sha256=digest(marker),files=len(plan['files']),copied_bytes=plan['total_bytes'])


def preflight():
 """Synthetic closed-file publication and faults, entirely /tmp."""
 from io import BytesIO
 import numpy as np
 folder=Path(tempfile.mkdtemp(prefix='natural-fit-publication-preflight-',dir='/tmp'));out=folder/'out';out.mkdir();checks=[]
 def check(n,v=True):require(v,n);checks.append(n)
 def fails(fn):
  try:fn()
  except RuntimeError:return True
  return False
 source=folder/'trainer.py';source.write_text('# Synthetic trainer\n');auditor=folder/'audit.py';auditor.write_text('# Synthetic auditor\n')
 for n in ARTIFACTS:atomic_json(out/n,dict(synthetic=True)) if n.endswith('.json') else (out/n).write_text('Synthetic\n')
 records=[dict(record=r,validation=dict(synthetic=True)) for r in RECORDS];atomic_json(out/'records.json',records)
 selected=dict(candidate='new-max33',promoted=False,primary_changed=False,tests_read=False,no_fallback=True,new_independent_confirmation_required=True);atomic_json(out/'selection.json',selected)
 with (out/'validation-metrics.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['record','group']);writer.writeheader();writer.writerows(dict(record=r,group=g) for r in RECORDS for g in GROUPS+['pooled_descriptive_only'])
 for arm in ['new','matched']:
  for n in ['models','training','candidates']:(out/n).mkdir(exist_ok=True)
  arrays={k:np.zeros(1,np.float32) for k in ['mean','scale','weight::encoder.0.weight','weight::encoder.0.bias','weight::head.weight','weight::head.bias']};arrays['metadata']=np.array('{}')
  np.savez(out/'models'/f'{arm}.npz',**arrays);atomic_json(out/'training'/f'{arm}.json',dict(history='Synthetic only'))
  atomic_json(out/'candidates'/f'{arm}.json',dict(model_sha256=digest(out/'models'/f'{arm}.npz'),training_sha256=digest(out/'training'/f'{arm}.json')))
 complete=dict(source_sha256=digest(source),tests_read=False,primary_changed=False,new_independent_confirmation_required=True,
    artifact_sha256={n:digest(out/n) for n in ARTIFACTS},candidate_sha256={a:digest(out/'candidates'/f'{a}.json') for a in ['new','matched']});atomic_json(out/'complete.json',complete)
 atomic_json(out/'status.json',dict(stage='validation_complete'))
 audit=dict(stage='actual_completed_natural_fit_development_audit',audit_passed=True,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,
    trainer_source_sha256=digest(source),trainer_protocol_sha256=digest(out/'protocol.json'),auditor_source_sha256=digest(auditor),trainer_complete_sha256=digest(out/'complete.json'),
    trainer_selection_sha256=digest(out/'selection.json'),trainer_records_sha256=digest(out/'records.json'),csv_primary_group_rows=88,csv_descriptive_pool_rows=8,
    trainer_model_record_sha256={a:digest(out/'candidates'/f'{a}.json') for a in ['new','matched']},model_sha256={a:digest(out/'models'/f'{a}.npz') for a in ['new','matched']},provenance_sha256={})
 ap=folder/'actual-audit.json';atomic_json(ap,audit)
 ctx=SimpleNamespace(out=out,dest=folder/'release',source=source,source_sha=digest(source),protocol_sha=digest(out/'protocol.json'),auditor=auditor,auditor_sha=digest(auditor),audit=ap,evidence=[],helper=Path(__file__).resolve())
 class ForbiddenRead(LookupError):pass
 def forbidden(*a,**k):raise ForbiddenRead('Unexpected formal reader')
 with patch(__name__+'.processes',return_value=[1]),patch(__name__+'.read',side_effect=forbidden):check('Livejob gate before result read',fails(lambda:build_plan(ctx)))
 with patch(__name__+'.processes',return_value=[]):
  plan=build_plan(ctx);check('Dryrun has no destination mutation and excludes raw payloads',not ctx.dest.exists() and plan['required_files_complete'] and all(e['kind']!='prediction' for e in plan['files']))
  for name in ['features.f16','new-probabilities.npz','.run.lock','portable-sample.npz']:
   path=folder/name;path.write_bytes(b'no raw payload');check('Exclude '+name,eligible(path)[0] is False)
  large=folder/'oversize.json'
  with large.open('wb') as f:f.truncate(MAX_FILE)
  check('Exclude50MiB without hashing/copy',eligible(large)==(False,'at_least_50MiB'))
  symlink=folder/'link.json';symlink.symlink_to(ap);check('Exclude symlink',eligible(symlink)==(False,'symlink'))
  pp=folder/'reviewed.json';atomic_json(pp,plan);result=apply_plan(ctx,plan,digest(pp),pp)
  check('Synthetic copy is bounded/hashchecked and complete only after closure',Path(result['manifest']).is_file() and result['files']==len(plan['files']))
  check('Synthetic apply is immutable/idempotent',apply_plan(ctx,plan,digest(pp),pp)==result)
  check('Wrong reviewed SHA rejected',fails(lambda:apply_plan(ctx,plan,'0'*64,pp)))
  p=ctx.dest/'models/new.npz';saved=p.read_bytes();p.write_bytes(b'tamper');check('Differing destination refused',fails(lambda:apply_plan(ctx,plan,digest(pp),pp)));p.write_bytes(saved)
  audit['audit_passed']=False;atomic_json(ap,audit);check('Failed independent audit blocks publication',fails(lambda:build_plan(ctx)))
 report=dict(stage='synthetic_publication_helper_preflight',checks=checks,checks_passed=len(checks),formal_results_or_models_read=False,formal_outputs_written=False,
    GPU_used=False,fixture=str(folder),helper_source_sha256=digest(__file__));atomic_json(folder/'report.json',report);print(json.dumps(dict(report=str(folder/'report.json'),report_sha256=digest(folder/'report.json'),**report)),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group();a.add_argument('--dry-run',action='store_true');a.add_argument('--apply',action='store_true');a.add_argument('--preflight',action='store_true')
 p.add_argument('--audit',type=Path,default=AUDIT);p.add_argument('--plan',type=Path,default=Path('/tmp/natural-fit-publication-plan.json'));p.add_argument('--plan-sha256');args=p.parse_args()
 if args.preflight:preflight()
 elif args.dry_run:
  require(args.plan.resolve().is_relative_to(Path('/tmp')),'Dry-run plan writes only /tmp.');value=build_plan(context(args.audit));atomic_json(args.plan,value)
  print(json.dumps(dict(plan=str(args.plan),plan_file_sha256=digest(args.plan),required_files_complete=value['required_files_complete'],files=len(value['files']),total_bytes=value['total_bytes'],excluded=value['excluded'])),flush=True)
 elif args.apply:
  require(args.plan_sha256 is not None and args.plan.resolve().is_relative_to(Path('/tmp')),'Apply requires exact reviewed /tmp plan and --plan-sha256.')
  print(json.dumps(apply_plan(context(args.audit),read(args.plan),args.plan_sha256,args.plan)),flush=True)
 else:p.print_help()

if __name__=='__main__':main()
