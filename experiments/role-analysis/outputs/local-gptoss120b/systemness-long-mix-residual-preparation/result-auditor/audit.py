"""Independent constant-long-weight result auditor; no production imports.

Default help; --preflight uses only synthetic /tmp fixtures. --actual is closed
until reviewed trainer SOURCE_SHA and PROTOCOL_SHA are hard-pinned. Only root may
launch actual after its exact training handle has returned terminal exit0.
The pinned, published 87c0 independent auditor supplies pure numerical routines;
its actual entrypoint, global verifiers, close(), and mutable globals are unused.
"""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

ROOT=Path('/root/prompt-injection-as-role-confusion')
HERE=ROOT/'experiments/role-analysis';BASE=HERE/'outputs/local-gptoss120b'
OUT=BASE/'systemness-long-mix-residual';CACHE=BASE/'systemness-expansion/cache/long-mix-residual'
SOURCE=HERE/'systemness_long_mix_residual.py'
SOURCE_SHA='230bc5e828a53e4b543b09a6c97693c0eec861ff01e6685424ec8d3f397f364d'
PROTOCOL_SHA='e636d28cd92449daa158cad8ebc8ba9c08060b50b5f1daac88199c6b34e9649f'
DESIGN=BASE/'systemness-long-mix-residual-preparation/root-prospective-design.json'
DESIGN_SHA='e2fb7dfb369c35f137ab23900b9eb611a8f33d9db8d8686ca4ee2c0961d36698'
MATH_SOURCE=BASE/'systemness-binary-residual-preparation/result-auditor/audit.py'
MATH_SHA='87c0e47c0ab9904fa3b77f477abe2bb3d437c66789773914d1b339d0194776c6'
PRIOR_OUT=BASE/'systemness-binary-residual'
PRIOR_AUDIT=PRIOR_OUT/'actual-audit/actual-development-audit.json'
PRIOR_AUDIT_SHA='2600d086c711e65bb320b70e66539fa775164cba8c192b84a2f9c52a75a9aa67'
PRIOR_MODEL_SHA='269c9d7e7b04248de8ea03a44a2db5484d6c793d3cfd3557d5fc39702b982779'
PRIOR_TAU=.7116969696969697
PRIOR_MAX33_TAU=.5453636363636364
MANIFEST_SHA='a7311c5f7ce1709c0c0947569fce4000c754e3e69a0490c8287490b44b93532b'
SLOTS_SHA='5c28655bec5eb6ba2dd90bd1422fb9cac920bb73f7fe2b7c729f5735c0915169'
ROLES=['system','user','cot','assistant','tool']
RECORDS=['new-oldmean','new-max33','matched-oldmean','matched-max33','oldhead-oldmean','oldhead-max33','ensemble-original','ensemble-max33','naturalCE-oldmean','naturalCE-max33','priorBinary-oldmean','priorBinary-max33']
ZERO_RECORDS=['zero-oldmean','zero-max33']
STRICT=['matched-max33','oldhead-max33','ensemble-max33','zero-max33','priorBinary-max33']
GUARDS=['matched-max33','oldhead-max33','ensemble-max33','oldhead-oldmean','ensemble-original','zero-oldmean','zero-max33','priorBinary-max33']
COLUMNS=['fit_position','old_bank','old_cache_ix','old_sample_ix','old_document_id','role','old_body_offset','half','new_bank','new_cache_ix','new_sample_ix','new_document_id','new_role','new_body_offset']


def require(value,message):
    if not value:raise RuntimeError(message)


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def close(got,want,label='',tolerance=1e-7):
    """Local recursive comparison; never changes the imported auditor's state."""
    if isinstance(want,dict):
        require(isinstance(got,dict) and set(got)==set(want),'Object keys differ: '+label)
        for k,v in want.items():close(got[k],v,label+'/'+str(k),tolerance)
    elif isinstance(want,(list,tuple)):
        require(isinstance(got,(list,tuple)) and len(got)==len(want),'Sequence differs: '+label)
        for i,v in enumerate(want):close(got[i],v,label+'/'+str(i),tolerance)
    elif want is None:require(got is None,'Undefined value replaced: '+label)
    elif isinstance(want,(bool,str,int)):require(got==want,'Discrete value differs: '+label)
    else:require(got is not None and np.isfinite(got) and np.isfinite(want) and abs(got-want)<=(2e-6 if label.endswith('/nll') else tolerance),'Numeric value differs: '+label)


def pure_math():
    require(digest(MATH_SOURCE)==MATH_SHA,'Published independent numerical library SHA changed.')
    spec=importlib.util.spec_from_file_location('long_mix_prior_independent_math',MATH_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.SOURCE_SHA=='fd49e6c8f199b9163d7ca2a5b7c4c66f2d0092b270d66a037394b0b2511cf9e9' and module.MAX_DIFFERENCE==0.,'Unexpected frozen mathematical module.')
    return module


def uniform(n,k):return np.linspace(0,n-1,k,dtype=int)


def replacement(fit,newfit,*,old_documents=132,new_documents=256,new_per_half=33):
    """Own metadata-only replay, independent of producer selection helpers."""
    require(fit.cache_ix.is_unique and fit.sample_ix.is_unique and fit.split.eq('fit').all(),'Old fit ownership or uniqueness failed.')
    old=fit.loc[fit.profile.eq('long-full') & fit.source_set.eq('long-full')].copy()
    require(old.document_id.nunique()==old_documents and len(old)==old_documents*1280 and old.wrapper.eq('native').all(),'Wrong original full sources.')
    require(newfit.split.eq('fit').all() and newfit.source_set.eq('long-augmentation').all() and newfit.profile.eq('long-full').all() and newfit.wrapper.eq('native').all()
        and newfit.document_id.nunique()==new_documents and len(newfit)==new_documents*1280 and newfit.cache_ix.is_unique and newfit.sample_ix.is_unique,'Wrong replacement fit sources.')
    require(not set(old.group_id)&set(newfit.group_id) and not set(old.sample_ix)&set(newfit.sample_ix),'Old/new source collision.')
    old['offset']=old.token_ix-old.groupby(['prompt_ix','message_ix']).token_ix.transform('min')
    old['slot']=pd.Series(np.arange(len(fit)),index=fit.cache_ix).loc[old.cache_ix].to_numpy()
    fresh=newfit.copy();fresh['offset']=fresh.body_offset
    expected_grid=uniform(1024,256);outgoing=[];incoming=[];retained=[]
    for pool,is_old in [(old,True),(fresh,False)]:
        require(pool.groupby('document_id').role.apply(lambda x:set(x)==set(ROLES)).all(),'Missing formal role.')
        document_offsets={}
        for (doc,role),frame in pool.groupby(['document_id','role'],sort=True):
            frame=frame.sort_values('offset');require(np.array_equal(frame.offset,expected_grid),'Frozen256 offsets changed.')
            chosen_offsets=[]
            for half in [0,1]:
                half_rows=frame.loc[frame.offset.between(half*512,half*512+511)].copy();require(len(half_rows)==128,'Unbalanced body halves.')
                chosen=np.zeros(128,bool);chosen[uniform(128,64 if is_old else new_per_half)]=True
                selected=half_rows.iloc[np.flatnonzero(chosen)].copy();selected['half']=half;chosen_offsets.extend(selected.offset)
                if is_old:
                    retained.append(selected);selected=half_rows.iloc[np.flatnonzero(~chosen)].copy();selected['half']=half;outgoing.append(selected)
                else:incoming.append(selected)
            document_offsets.setdefault(doc,tuple(chosen_offsets));require(document_offsets[doc]==tuple(chosen_offsets),'Five roles differ in selected body offsets.')
    old_rows=pd.concat(outgoing);new_rows=pd.concat(incoming);keep=pd.concat(retained)
    require(old_documents*64==new_documents*new_per_half,'Replacement bucket budgets differ.')
    rng=np.random.RandomState(20516086);pieces=[];buckets={}
    for role in ROLES:
        for half in [0,1]:
            a=old_rows.loc[old_rows.role.eq(role)&old_rows.half.eq(half)].sort_values('slot')
            b=new_rows.loc[new_rows.role.eq(role)&new_rows.half.eq(half)].sort_values(['document_id','offset','sample_ix'])
            require(len(a)==len(b)==old_documents*64,'Invalid role/half budget.');b=b.iloc[rng.permutation(len(b))]
            pieces.append(pd.DataFrame(dict(fit_position=a.slot.to_numpy(),old_bank='old',old_cache_ix=a.cache_ix.to_numpy(),old_sample_ix=a.sample_ix.to_numpy(),
                old_document_id=a.document_id.to_numpy(),role=a.role.to_numpy(),old_body_offset=a.offset.to_numpy(),half=half,new_bank='c4',new_cache_ix=b.cache_ix.to_numpy(),
                new_sample_ix=b.sample_ix.to_numpy(),new_document_id=b.document_id.to_numpy(),new_role=b.role.to_numpy(),new_body_offset=b.offset.to_numpy())))
            buckets[role+':'+str(half)]=len(a)
    result=pd.concat(pieces).sort_values('fit_position').reset_index(drop=True)[COLUMNS]
    require(result.fit_position.is_unique and result.new_cache_ix.is_unique and result.old_cache_ix.is_unique and result.new_sample_ix.is_unique,'Nonbijective replacement.')
    mapping=dict(zip(ROLES,range(5)));old_y=fit.role.map(mapping).to_numpy();new_y=old_y.copy();new_y[result.fit_position]=result.new_role.map(mapping)
    require(np.array_equal(old_y,new_y) and np.array_equal(result.role,result.new_role) and np.array_equal(result.half,result.old_body_offset.ge(512).astype(int))
        and np.array_equal(result.half,result.new_body_offset.ge(512).astype(int)),'Per-slot label or half mismatch.')
    summary=dict(seed=20516086,draw_once=True,torch_rng_consumed=False,rows=len(result),old_pool_documents=old_documents,new_pool_documents=new_documents,
        kept_old_full_rows=len(keep),selected_new_full_rows=len(new_rows),total_full_rows=len(keep)+len(new_rows),role_half_buckets=buckets,
        fit_position_sha256=array_sha(result.fit_position.to_numpy()),old_cache_ix_sha256=array_sha(result.old_cache_ix.to_numpy()),new_cache_ix_sha256=array_sha(result.new_cache_ix.to_numpy()),
        old_sample_ix_sha256=array_sha(result.old_sample_ix.to_numpy()),new_sample_ix_sha256=array_sha(result.new_sample_ix.to_numpy()),original_fit_labels_sha256=array_sha(old_y),
        replacement_labels_sha256=array_sha(old_y[result.fit_position.to_numpy()]),new_labels_sha256=array_sha(result.new_role.map(mapping).to_numpy()),kept_old_sources=old_documents,selected_new_sources=new_documents,
        new_role_rows_per_document=2*new_per_half,old_role_rows_per_document=128,matched_actual_clean_natural=True,matched_previous_binary_arm='new',body_position_histogram_matched=False)
    return result,summary


def slot_order_evidence(fit,natural,slots,*,production=True):
    """Features change by slot, original/natural order and full labels do not."""
    require(natural.split.eq('fit').all() and set(natural.role)<={'system','user','assistant'},'Non-fit or invented natural role.')
    require(not set(fit.sample_ix)&set(natural.sample_ix),'Natural namespace collision.')
    y=np.r_[fit.role.map(dict(zip(ROLES,range(5)))),natural.role.map(dict(zip(ROLES,range(5))))].astype(np.int64)
    new_y=y.copy();new_y[slots.fit_position]=slots.new_role.map(dict(zip(ROLES,range(5))))
    require(np.array_equal(y,new_y),'Total label sequence differs.')
    original_cache=fit.cache_ix.to_numpy();new_cache=original_cache.copy();new_cache[slots.fit_position]=slots.new_cache_ix
    banks=np.full(len(fit),'old',dtype='U3');banks[slots.fit_position]='c4'
    counts=dict(old=int(np.count_nonzero(banks=='old')),c4=int(np.count_nonzero(banks=='c4')),natural=len(natural))
    if production:require(len(y)==641493 and len(fit)==578090 and len(slots)==84480 and counts==dict(old=493610,c4=84480,natural=63403),'Production slot/order budget differs.')
    return dict(instances=len(y),old_fit_rows=len(fit),natural_rows=len(natural),replacement_rows=len(slots),new_arm_bank_rows=counts,
        matched_arm_bank_rows=dict(old=len(fit),c4=0,natural=len(natural)),matched_previous_binary_arm='new',
        five_role_labels_identical_by_slot=True,all_epoch_batch_labels_identical_for_same_permutation=True,all_labels_sha256=array_sha(y),
        original_fit_cache_ix_sha256=array_sha(original_cache),new_fit_cache_ix_sha256=array_sha(new_cache),new_fit_bank_sha256=array_sha(banks),
        unchanged_natural_sample_ix_sha256=array_sha(natural.sample_ix.to_numpy()),matched_not_prior_binary_matched_role_draw=True)


def select_records(math,records,zero,legacy):
    require(len(records)==12 and {r['record'] for r in records}==set(RECORDS) and len(zero)==2 and {r['record'] for r in zero}==set(ZERO_RECORDS),'Expected12 formal plus2 separate zero records.')
    index={r['record']:r['validation'] for r in records+zero};candidate=index['new-max33'];minimum=candidate['min33'];reasons=[];guards={};gains={}
    if minimum is None or not np.isfinite(minimum) or minimum<.9:reasons.append('absolute_min33_below_90_percent')
    for name in GUARDS:
        g=math.guard(candidate,index[name]);guards[name]=g;reasons.extend(name+':'+r for r in g['rejection_reasons'])
        if name in STRICT:
            gain=minimum-index[name]['min33'] if minimum is not None and index[name]['min33'] is not None else None;gains[name]=gain
            if gain is None or not np.isfinite(gain) or gain<=1e-12:reasons.append(name+':min33_not_strictly_improved')
    gpu=math.guard(candidate,legacy,True);guards['original-savedGPU']=gpu;reasons.extend('original-savedGPU:'+r for r in gpu['rejection_reasons'])
    return dict(candidate='new-max33',promoted=not reasons,no_fallback=True,min33_gains=gains,guards=guards,rejection_reasons=reasons,naturalCE_is_diagnostic_only=True,
        priorBinary_max33_is_strict_and_guard=True,primary_changed=False,tests_read=False,new_independent_confirmation_required=True)


def verify_csv(frame,records,names):
    require(len(records)==len(names) and {r['record'] for r in records}==set(names),'CSV expected record coverage differs.')
    expected={(r['record'],g):v for r in records for g,v in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()}
    require(len(frame)==12*len(names) and not frame.duplicated(['record','group']).any() and set(zip(frame.record,frame.group))==set(expected),'CSV row identities differ.')
    for row in frame.to_dict('records'):
        value=expected[row['record'],row['group']]
        for k,v in value.items():
            if not isinstance(v,(dict,list)):close(None if pd.isna(row[k]) else row[k],v,'csv/'+k)
        for role in ROLES:close(None if pd.isna(row['role_recall_'+role]) else row['role_recall_'+role],value['recall'][role],'csv/role/'+role)


def replication_evidence(math,current,prior,p,current_training,prior_training,*,signature=None,current_model_sha=None,training_sha=None):
    """Raw model arrays and each full-bank p, no producer model loader."""
    require(set(current)==set(prior),'Replication parameter sets differ.');weights={}
    for name,a in current.items():
        b=prior[name];require(a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Replication parameter shape/finiteness differs.')
        difference=np.abs(a.astype(np.float64)-b.astype(np.float64));weights[name]=dict(array_bitexact=bool(np.array_equal(a.view(np.uint32),b.view(np.uint32))),maximum_absolute_difference=float(difference.max()),mean_absolute_difference=float(difference.mean()),current_sha256=array_sha(a),prior_sha256=array_sha(b))
    drift={b:dict(original_oldhead_tau=math.probability_drift(p['priorBinary'][b],p['matched'][b],math.CURRENT_TAU),prior_binary_original_tau=math.probability_drift(p['priorBinary'][b],p['matched'][b],PRIOR_TAU)) for b in ['old','c4','public']}
    require(all(v['original_oldhead_tau']['maximum_probability_difference']<5e-4 for v in drift.values()),'Matched replication differs by >=5e-4; do not attribute drift to data.')
    return dict(signature=signature,matched_data='EXACT previous binary NEW original578090+actual63403 natural, same order and labels.',prior_model_sha256=PRIOR_MODEL_SHA,
        current_model_sha256=current_model_sha,prior_training_sha256=(training_sha or {}).get('prior'),current_training_sha256=(training_sha or {}).get('current'),weights=weights,
        all_weight_arrays_bitexact=all(v['array_bitexact'] for v in weights.values()),probability_drift=drift,
        all_probability_arrays_bitexact=all(np.array_equal(p['matched'][b].view(np.uint32),p['priorBinary'][b].view(np.uint32)) for b in ['old','c4','public']),
        budget_comparison={name:{k:v[k] for k in ['best_epoch','best_steps','epochs','actual_steps','actual_instances']} for name,v in [('current',current_training),('prior',prior_training)]},
        technical_tolerance=5e-4,numerical_drift_is_not_source_effect=True,tests_read=False)


def require_pins(source_sha,protocol_sha):
    require(source_sha is not None and protocol_sha is not None,'Reviewed final trainer source/protocol not pinned: actual forbidden.')


def terminal_gate(*,out=OUT,active=None,source_sha=None,protocol_sha=None):
    source_sha=SOURCE_SHA if source_sha is None else source_sha;protocol_sha=PROTOCOL_SHA if protocol_sha is None else protocol_sha
    require_pins(source_sha,protocol_sha)
    if active is None:
        active=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
            try:args=(entry/'cmdline').read_bytes().split(b'\0')
            except(FileNotFoundError,PermissionError,ProcessLookupError):continue
            if any(Path(a.decode(errors='replace')).name in {'systemness_long_mix_residual.py','systemness_binary_residual.py','systemness_natural_dev_extract.py'} for a in args if a):active.append(int(entry.name))
    require(not active,'Trainer/extractor still live: no partial reads.')
    require((out/'complete.json').is_file() and (out/'status.json').is_file(),'Formal completion/status absent.')
    require(read(out/'status.json')['stage']=='validation_complete','Trainer not terminal complete.')


def raw_evidence(saved,computed,label):
    """Local comparison avoids the previous auditor's mutable close counter."""
    t=np.asarray(computed['five_class_transition']);expected=dict(tokens=computed['rows'],raw_non_system_sha256=computed['non_system_raw_after_sha256'],
        base_raw_non_system_sha256=computed['non_system_raw_before_sha256'],raw_non_system_argmax_changes=computed['conditional_raw_argmax_changes'],
        delta_min=computed['delta_min'],delta_max=computed['delta_max'],delta_mean=computed['delta_mean'],base_to_residual_p_argmax=t.tolist(),
        p_conditional_non_system_argmax_changes=computed['p_conditional_non_system_argmax_changes'],p_non_system_tie_rows=computed['p_non_system_tie_rows'],
        p_non_system_underflow_values=computed['p_non_system_underflow_values'],CoT_to_assistant_p_argmax=int(t[2,3]),non_system_to_system_p_argmax=int(t[1:,0].sum()),
        system_to_non_system_p_argmax=int(t[0,1:].sum()),non_system_to_different_non_system_p_argmax=int(t[1:,1:].sum()-np.trace(t[1:,1:])),
        metric_convention='Unmodified savedF32 p.argmax; raw-tail bitidentity does not guarantee p rounding preserves non-system argmax.',
        transition_definition='Rows index original base p.argmax; columns index residual p.argmax, regardless of ground-truth role. CoT_to_assistant is a newly introduced winner transition, not total truth-CoT misclassification.')
    require(expected['non_system_to_different_non_system_p_argmax']==0 and expected['raw_non_system_argmax_changes']==0
        and expected['raw_non_system_sha256']==expected['base_raw_non_system_sha256'],'Raw/p non-system invariance failed.')
    close(saved,expected,label)


def raw_state(path):
    with np.load(path,allow_pickle=False) as z:return {k[8:]:z[k].copy() for k in z.files if k.startswith('weight::')}


def verified_banks(math,static,bind):
    """Explicit finite prepared banks, no production verifier or inventory scan."""
    folders={'old':BASE/'systemness-head-features','c4':BASE/'systemness-long-augmentation','public':BASE/'systemness-natural-dev-features'}
    counts={'old':755282,'c4':389120,'public':80116};tables={};banks={};heads=[458,462,2045,460,459,2047,456,997]
    require(static['selected_head_ids']==heads,'Ranked selected head order changed.')
    for bank,folder in folders.items():
        path=folder/'extraction-complete.json';bind(path);complete=read(path);cache=math.BANK_PATHS[bank]
        require(complete['selected_head_ids']==heads and complete['all_finite'] is True and complete['all_rows_written'] is True
            and complete['rows']==counts[bank] and complete['dimensions']==512,'Prepared feature bank scope/completeness changed.')
        feature=cache/'features.f16';require(feature.stat().st_size==counts[bank]*512*2,'Feature F16 size changed.')
        require(static['feature_files_sha256'][str(feature)]==complete['features_sha256'],'Feature metadata/static binding differs.')
        bind(feature,complete['features_sha256']);bind(cache/'rows.parquet',complete['rows_sha256']);tables[bank]=pd.read_parquet(cache/'rows.parquet')
        banks[bank]=np.memmap(feature,mode='r',dtype=np.float16,shape=(counts[bank],512))
    fit,rows,natural,partition=math.source_partitions(tables['old'],tables['c4'],tables['public']);close(partition,static['original_partitions'],'original partition')
    newfit=tables['c4'].loc[tables['c4'].split.eq('fit')].copy().reset_index(drop=True)
    require(len(newfit)==327680 and newfit.document_id.nunique()==256 and newfit.document_id.min()>=6000000 and newfit.sample_ix.min()>=6000000000000
        and not set(newfit.group_id)&set(rows['c4'].group_id),'New source fit/validation namespace or membership changed.')
    return fit,rows,natural,newfit,partition,banks


def actual(output):
    terminal_gate() # pins/no-live/status first, before slots, p or model opens
    output=Path(output);require(output.resolve().is_relative_to(Path('/tmp')),'Auditor output must be /tmp.')
    math=pure_math();old_globals=(math.RECORDS.copy(),math.GUARD_CONTROLS.copy(),math.STRICT_CONTROLS.copy(),math.MAX_DIFFERENCE)
    bindings={};probability_bindings={}
    def bind(path,want=None):
        path=Path(path);require(path.is_file(),'Missing closed artifact: '+str(path));got=digest(path)
        if want is not None:require(got==want,'Artifact SHA changed: '+str(path))
        bindings[str(path.resolve())]=got;return got
    def p_read(path,want,frame):
        probability_bindings[str(Path(path).resolve())]=bind(path,want);return math.read_probability(path,want,frame)
    bind(__file__);bind(MATH_SOURCE,MATH_SHA);bind(SOURCE,SOURCE_SHA);bind(DESIGN,DESIGN_SHA);bind(OUT/'protocol.json',PROTOCOL_SHA);bind(OUT/'status.json')
    complete_sha=bind(OUT/'complete.json');complete=read(OUT/'complete.json');plan_sha=bind(OUT/'search-plan.json');plan=read(OUT/'search-plan.json')
    declaration=read(OUT/'protocol.json');proto=declaration['protocol'];static=declaration['static'];sig=plan['signature'];design=read(DESIGN)
    artifacts={'protocol.json','search-plan.json','replacement-manifest.json','controls.json','zero-controls.json','replication.json','records.json','selection.json','validation-metrics.csv','zero-control-metrics.csv','source-counts.json','report.txt'}
    require(complete['signature']==sig and complete['source_sha256']==SOURCE_SHA and complete['tests_read'] is False and complete['primary_changed'] is False
        and complete['new_independent_confirmation_required'] is True and set(complete['artifact_sha256'])==artifacts and set(complete['candidate_sha256'])=={'new','matched'},'Completion ownership/artifact scope changed.')
    for name,want in complete['artifact_sha256'].items():bind(OUT/name,want)
    require(plan['protocol']==proto and static['source_sha256']==SOURCE_SHA and static['design_sha256']==DESIGN_SHA and static['validation_probability_arrays_read'] is False
        and static['global_inventory_scanned'] is False and proto['tests_read'] is False and proto['primary_changed'] is False and proto['records']==RECORDS
        and proto['groups']==math.GROUPS and proto['feasible']==math.FEASIBLE and proto['max_FPR']==.05 and proto['zero_controls']==ZERO_RECORDS,'Prospective protocol changed.')
    expected_sig=dict(source_sha256=SOURCE_SHA,protocol_sha256=PROTOCOL_SHA,static_sha256=canonical(static),selected_head_ids=static['selected_head_ids'],
        standardization_sha256=static['standardization_sha256'],base_model_sha256=math.CURRENT_SHA,embedded_base_weight_sha256=static['embedded_base_weight_sha256'],
        feature_files_sha256=static['feature_files_sha256'],replacement_slots_sha256=static['replacement_slots_sha256']);close(sig,expected_sig,'signature')
    close(proto['neural'],dict(seed=20261021,residual_hidden=32,dropout=.1,optimizer='AdamW',lr=.001,weight_decay=.001,clip=5.,batch=4096,instances=641493,steps=157,last_batch=2517,max_epochs=45,min_epochs=10,patience=8,improvement=1e-5),'protocol/neural')
    require(proto['CPU']==dict(threads=4,float32_matmul_precision='highest',dtype='float32',chunk=4096) and proto['frozen_base']['model_sha256']==math.CURRENT_SHA
        and proto['promotion']['strict_controls']==STRICT and proto['promotion']['guard_controls']==GUARDS and proto['promotion']['absolute_min33']==.9,'Numerics or5strict/8guards changed.')
    close(proto['replacement'],dict(seed=20516086,draw_once=True,old_full_rows=168960,old_kept=84480,replaced=84480,new_fit_sources=256,new_per_source_role=66,old_per_source_role=128,role_half_buckets=10,per_bucket=8448,torch_rng_consumed=False,compact_manifest='replacement-slots.parquet'),'protocol/replacement')
    require(static['packages']=={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','pyarrow','scikit-learn']},'Portable runtime package versions changed.')
    for name,want in static['dependencies_sha256'].items():bind(HERE/name,want)
    for path,want in static['closed_files_sha256'].items():bind(path,want)
    require(static['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA and static['prior_binary_model_sha256']==PRIOR_MODEL_SHA,'Prior binary closed certificate/model pin changed.')
    bind(PRIOR_AUDIT,PRIOR_AUDIT_SHA);prior=read(PRIOR_AUDIT);prior_complete=read(PRIOR_OUT/'complete.json');prior_protocol=read(PRIOR_OUT/'protocol.json')
    require(prior['stage']=='actual_completed_binary_residual_development_audit' and prior['audit_passed'] is True and prior['test_data_or_results_read'] is False
        and prior['trainer_source_sha256']==math.SOURCE_SHA and prior['trainer_protocol_sha256']==math.PROTOCOL_SHA and prior['model_sha256']['new']==PRIOR_MODEL_SHA,'Prior audited binary trial identity changed.')
    for filename,field in [('complete.json','trainer_complete_sha256'),('protocol.json','trainer_protocol_sha256'),('selection.json','trainer_selection_sha256'),('records.json','trainer_records_sha256')]:bind(PRIOR_OUT/filename,prior[field])
    for name,want in prior_complete['artifact_sha256'].items():bind(PRIOR_OUT/name,want)
    for name,want in prior_complete['candidate_sha256'].items():bind(PRIOR_OUT/'candidates'/f'{name}.json',want)
    for path,want in prior['provenance_sha256'].items():bind(path,want)
    for path,want in prior['probability_bindings'].items():probability_bindings[str(Path(path).resolve())]=bind(path,want)
    bind(prior['auditor_source_path'],prior['auditor_source_sha256'])
    fit,rows,natural,newfit,partition,banks=verified_banks(math,static,bind)
    for name,frame in [('fit',fit),('old',rows['old']),('c4',rows['c4']),('natural-fit',natural),('public',rows['public'])]:
        require(array_sha(frame.sample_ix.to_numpy())==static['sample_ix_sha256'][name]==design['data']['original_reference_source_sample_sha256'][name],'Original reference sample order changed: '+name)
    slots,summary=replacement(fit,newfit);slot_path=CACHE/'replacement-slots.parquet';require(static['replacement_slots_sha256']==SLOTS_SHA and static['replacement_manifest_sha256']==MANIFEST_SHA,'Root-declared immutable replacement pins changed.');bind(slot_path,SLOTS_SHA)
    pd.testing.assert_frame_equal(pd.read_parquet(slot_path),slots,check_exact=True)
    descriptor=read(OUT/'replacement-manifest.json');close(descriptor,summary|dict(slots_path=str(slot_path),slots_sha256=static['replacement_slots_sha256']),'replacement descriptor')
    require(bind(OUT/'replacement-manifest.json')==static['replacement_manifest_sha256'],'Replacement descriptor/protocol binding changed.')
    slot_proof=slot_order_evidence(fit,natural,slots);require(array_sha(fit.sample_ix.to_numpy())==prior_protocol['static']['sample_ix_sha256']['fit']
        and array_sha(natural.sample_ix.to_numpy())==prior_protocol['static']['sample_ix_sha256']['natural-fit'],'Matched membership/order differs from prior binary NEW.')
    scaler_path=BASE/'systemness-head-probes/standardization.npz';scaler_sha=bind(scaler_path,static['standardization_sha256'])
    with np.load(scaler_path,allow_pickle=False) as z:mean,scale=z['mean'].copy(),z['scale'].copy()
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32 and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Frozen scaler malformed.')
    bind(static['base_model_path'],math.CURRENT_SHA);_,base=math.model_arrays(static['base_model_path'],mean,scale)
    require(math.weight_state_sha(base,True)==static['embedded_base_weight_sha256'] and array_sha(mean)==static['mean_sha256'] and array_sha(scale)==static['scale_sha256'],'Frozen base/scaler arrays changed.')
    initial,initial_sha=math.initial_binary_state(base);require(initial_sha==plan['initial_weight_sha256']==static['initial_weight_sha256'],'Full base+zero-last initialization changed.')
    probs={};budgets={};model_hashes={};replays={};training_sha={}
    def load_ps(name,hashes):
        require(set(hashes)==set(rows),'Probability cohort coverage changed.');return {b:p_read(CACHE/f'{name}-{b}-validation.npz',hashes[b],frame) for b,frame in rows.items()}
    def expected_meta(arm):
        return dict(kind='binary-residual',arm=arm,dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=ROLES,base_model_sha256=math.CURRENT_SHA,
            embedded_base_weight_sha256=static['embedded_base_weight_sha256'],mean_sha256=static['mean_sha256'],scale_sha256=static['scale_sha256'],selected_head_ids=sig['selected_head_ids'],
            signature=sig,source_sha256=SOURCE_SHA,plan_sha256=plan_sha,initial_weight_sha256=initial_sha,standardization_sha256=scaler_sha,tests_read=False)
    def replay(name,path,sha,rawproof,meta):
        model_hashes[name]=bind(path,sha);actual_meta,bb,rr=math.binary_model_arrays(path,base,mean,scale)
        for k,v in meta.items():close(actual_meta[k],v,'portable/'+name+'/'+k)
        proofs={}
        for b,frame in rows.items():
            computed,oldp,inv=math.residual_replay(bb,rr,mean,scale,banks[b],frame);difference=float(np.abs(computed-probs[name][b]).max())
            require(difference<5e-6,'Independent rawF16/base/residual/p replay mismatch: '+name+'/'+b);raw_evidence(rawproof[b],inv,name+'/'+b)
            proofs[b]=dict(tokens=len(frame),maximum_probability_difference=difference,independent_probability_sha256=array_sha(computed),saved_probability_sha256=array_sha(probs[name][b]),base_probability_sha256=array_sha(oldp),raw_invariance=inv)
        replays[name]=proofs;return rr
    for arm in ['new','matched']:
        cp=OUT/'candidates'/f'{arm}.json';bind(cp,complete['candidate_sha256'][arm]);candidate=read(cp);tp=OUT/'training'/f'{arm}.json';training_sha[arm]=bind(tp,candidate['training_sha256']);training=read(tp)
        amounts=math.history_audit(training['history'],training['best_epoch']);budgets[arm]=amounts
        for obj in [training,candidate]:
            require(obj['arm']==arm and obj['signature']==sig and obj['plan_sha256']==plan_sha and obj['initial_weight_sha256']==initial_sha,'Training ownership/init differs.')
            for k,v in amounts.items():close(obj[k],v,arm+'/budget/'+k)
        require(candidate['source_sha256']==SOURCE_SHA and candidate['tests_read'] is False and candidate['model_sha256']==training['model_sha256']
            and candidate['gpu_old_probability_sha256']==training['gpu_old_probability_sha256'] and training['new_C4_predictions_read'] is False and training['public_validation_predictions_read'] is False
            and training['embedded_base_weight_sha256']==static['embedded_base_weight_sha256'] and training['base_eval_all_epochs'] is True and training['base_gradient_observations']==0
            and training['optimizer_only_residual'] is True and candidate['cuda_visible_devices'] in ['0','1'] and 'A40' in candidate['cuda_device_name'],'Frozen base/best-before-new/device evidence changed.')
        math.parity_proof(candidate['portable_evidence'],natural);probs[arm]=load_ps(arm,candidate['probability_sha256'])
        gp=p_read(CACHE/f'{arm}-gpu-old-validation.npz',candidate['gpu_old_probability_sha256'],rows['old']);gt=math.legacy_mean(rows['old'],gp)
        gv=math.evaluate(rows,dict(probs[arm],old=gp),gt);close(math.old_validation_record(gv,rows['old']),candidate['old_early_stop_validation'],arm+'/earlystop')
        close(gv['old5_mean_f1'],training['history'][training['best_epoch']-1]['old5_mean_f1'],arm+'/best/history')
        dp=math.probability_drift(gp,probs[arm]['old'],math.CURRENT_TAU);close(dp,candidate['full_old_GPU_CPU_drift'],arm+'/GPU_CPU');require(dp['maximum_probability_difference']<5e-4,'Full bestGPU/CPU drift exceeds gate.')
        replay(arm,OUT/'models'/f'{arm}.npz',candidate['model_sha256'],candidate['raw_invariance'],expected_meta(arm)|dict(best_epoch=amounts['best_epoch'],best_steps=amounts['best_steps']))
    controls=read(OUT/'controls.json');require(controls['signature']==sig and controls['prior_actual_audit_sha256']==PRIOR_AUDIT_SHA and controls['new_bank_audit_sha256']==static['new_bank_audit_sha256']
        and controls['reference_probabilities']==static['reference_probabilities'] and controls['reference_probabilities_recomputed'] is False and controls['naturalCE_diagnostic_only'] is True
        and controls['priorBinary_max33_is_strict_and_guard'] is True and controls['tests_read'] is False,'Inherited controls closure changed.')
    require(set(static['reference_probabilities'])=={'oldhead','ensemble','naturalCE','priorBinary'},'Inherited physical control scope differs.')
    for name,items in static['reference_probabilities'].items():
        require(set(items)==set(rows),'Inherited bank scope differs.');probs[name]={}
        for b,item in items.items():
            require(item['sample_ix_sha256']==array_sha(rows[b].sample_ix.to_numpy()) and prior['probability_bindings'][str(Path(item['path']).resolve())]==item['sha256'],'Inherited reference not bound by prior actual certificate.')
            probs[name][b]=p_read(item['path'],item['sha256'],rows[b])
    expected_physical=dict(oldhead=math.CURRENT_SHA,ensemble=static['prior_model_sha256'],naturalCE='204f4a9dbfb7855460ccb210ca703aca721b974a87059be9134f55c4f7b5cba5',priorBinary=PRIOR_MODEL_SHA)
    close(controls['physical_model_sha256'],expected_physical,'physical controls');model_hashes.update(expected_physical)
    # Prior87c certificate already proves fixed4 F64sum/4=>F32 and original
    # oldhead alias. Recheck its attestation and every referenced NPZ byte SHA.
    require(prior['fixed_four_member_ensemble'] is True and prior['original_savedGPU_NPZ_and_threshold_unchanged'] is True,'Inherited arithmetic/control attestation absent.')
    ref=static['old_reference'];gpu={b:p_read(ref[k+'_path'],ref[k+'_sha256'],rows[b]) for b,k in [('old','old'),('c4','new')]}
    legacy=math.evaluate(rows,dict(gpu,public=probs['oldhead']['public']),math.CURRENT_TAU);close(controls['original_savedGPU_validation'],legacy,'original savedGPU')
    prior_model=Path(static['prior_binary_model_path']);bind(prior_model,PRIOR_MODEL_SHA);math.binary_model_arrays(prior_model,base,mean,scale)
    prior_tp=PRIOR_OUT/'training/new.json';current_tp=OUT/'training/matched.json';oldtraining=read(prior_tp);matched_training=read(current_tp)
    rp=replication_evidence(math,raw_state(OUT/'models/matched.npz'),raw_state(prior_model),probs,matched_training,oldtraining,signature=sig,
        current_model_sha=model_hashes['matched'],training_sha=dict(current=bind(current_tp),prior=bind(prior_tp)))
    close(read(OUT/'replication.json'),rp,'matched replication')
    zero=read(OUT/'zero-controls.json');require(zero['signature']==sig and zero['both_arm_training_sha256']==training_sha and zero['original_threshold']==math.CURRENT_TAU
        and zero['trained_record'] is False and zero['tests_read'] is False and zero['model_path']==str(OUT/'models/zero.npz'),'Zero samepath/both-frozen closure changed.')
    math.parity_proof(zero['portable_evidence'],natural);probs['zero']=load_ps('zero',zero['probability_sha256']);zr=replay('zero',OUT/'models/zero.npz',zero['model_sha256'],zero['raw_invariance'],expected_meta('zero')|dict(zero_residual=True))
    require(np.array_equal(zr['encoder.0.weight'],initial['correction.0.weight']) and np.array_equal(zr['encoder.0.bias'],initial['correction.0.bias']) and not np.any(zr['head.weight']) and not np.any(zr['head.bias']),'Samepath zero residual initialization changed.')
    for b in rows:
        inv=replays['zero'][b]['raw_invariance'];require(inv['delta_min']==inv['delta_max']==inv['delta_mean']==0,'Zero produced nonzero delta.')
        d=math.probability_drift(probs['oldhead'][b],probs['zero'][b],math.CURRENT_TAU);close(d,zero['oldhead_commonCPU_drift'][b],'zero/drift/'+b);require(d['maximum_probability_difference']<5e-4,'Samepath zero drift exceeds gate.')
    zt,zproof,zannex=math.direct_grid(rows,probs['zero']);zero_records=[dict(record='zero-oldmean',validation=math.evaluate(rows,probs['zero'],math.CURRENT_TAU)),dict(record='zero-max33',validation=math.evaluate(rows,probs['zero'],zt))]
    close(zero['records'],{r['record']:r['validation'] for r in zero_records},'zero/records');close(zero['max33_full_grid_proof'],zproof,'zero/fullgrid')
    saved=read(OUT/'records.json');index={r['record']:r for r in saved};require(len(saved)==12 and set(index)==set(RECORDS),'Formal12 records incomplete.')
    records=[];grid_proofs=dict(zero=dict(proof=zproof,annex=zannex))
    for name in RECORDS:
        model,objective=name.split('-');p=probs[model]
        if objective=='max33':tau,proof,annex=math.direct_grid(rows,p);grid_proofs[model]=dict(proof=proof,annex=annex)
        else:
            tau={'oldhead':math.CURRENT_TAU,'ensemble':math.ENSEMBLE_TAU,'naturalCE':.7772222222222223,'priorBinary':PRIOR_TAU}.get(model)
            if tau is None:tau=math.legacy_mean(rows['old'],p['old'])
            proof=None
        record=dict(record=name,model=model,objective=objective,validation=math.evaluate(rows,p,tau),full_grid_proof=proof,diagnostic=name!='new-max33',tests_read=False)
        if name=='naturalCE-max33':require(tau==.6008080808080808,'Inherited CE max33 changed.')
        if name=='priorBinary-max33':require(tau==PRIOR_MAX33_TAU and record['validation']['min33']==.8990885416666666,'Inherited priorBinary calibration changed.')
        close(index[name],record,name);records.append(record)
    selected=select_records(math,records,zero_records,legacy);published=read(OUT/'selection.json')
    for k,v in selected.items():close(published[k],v,'selection/'+k)
    require(published['source_sha256']==SOURCE_SHA and published['plan_sha256']==plan_sha and published['records_sha256']==complete['artifact_sha256']['records.json']
        and published['controls_sha256']==complete['artifact_sha256']['controls.json'] and published['zero_controls_sha256']==complete['artifact_sha256']['zero-controls.json']
        and published['replication_sha256']==complete['artifact_sha256']['replication.json'] and published['model_path']==str(OUT/'models/new.npz')
        and published['model_sha256']==model_hashes['new'] and published['system_threshold']==index['new-max33']['validation']['system_threshold'],'Selected model/reference identity changed.')
    verify_csv(pd.read_csv(OUT/'validation-metrics.csv'),records,RECORDS);verify_csv(pd.read_csv(OUT/'zero-control-metrics.csv'),zero_records,ZERO_RECORDS)
    counts=read(OUT/'source-counts.json');oldcounts=read(PRIOR_OUT/'source-counts.json')
    extra=dict(experiment='constant-long-weight-source-replacement-residual',same_existing_rows=False,same_validation_rows=True,old_full_rows=168960,kept_old_full_rows=84480,
        new_full_rows=84480,total_full_fraction=168960/641493,new_arm_bank_rows=slot_proof['new_arm_bank_rows'],matched_arm_bank_rows=slot_proof['matched_arm_bank_rows'],replacement=descriptor,limitation=proto['limitation'])
    close(counts,oldcounts|extra,'source counts');require(counts['per_epoch']==641493 and counts['steps']==157 and counts['last_batch']==2517,'Training sample budget differs.')
    source_counts={g:dict(bank=b,tokens=len(ix),sources=int(rows[b].iloc[ix].group_id.nunique()),role_counts={r:int(rows[b].iloc[ix].role.eq(r).sum()) for r in ROLES},sample_ix_sha256=array_sha(rows[b].iloc[ix].sample_ix.to_numpy())) for g,(b,ix) in math.group_indices(rows).items()}
    terminal_gate();require(old_globals==(math.RECORDS,math.GUARD_CONTROLS,math.STRICT_CONTROLS,math.MAX_DIFFERENCE),'Frozen pure-math module state was changed.')
    for path,want in bindings.items():require(digest(path)==want,'Bound artifact changed during audit: '+path)
    report=dict(stage='actual_completed_long_mix_residual_development_audit',audit_passed=True,auditor_source_path=str(Path(__file__).resolve()),auditor_source_sha256=digest(__file__),
        pure_math_source_path=str(MATH_SOURCE),pure_math_source_sha256=MATH_SHA,trainer_source_sha256=SOURCE_SHA,trainer_complete_sha256=complete_sha,trainer_protocol_sha256=PROTOCOL_SHA,
        trainer_selection_sha256=bindings[str((OUT/'selection.json').resolve())],trainer_records_sha256=bindings[str((OUT/'records.json').resolve())],
        trainer_model_record_sha256={a:bindings[str((OUT/'candidates'/f'{a}.json').resolve())] for a in ['new','matched']},prior_actual_audit_sha256=PRIOR_AUDIT_SHA,
        model_sha256=model_hashes,provenance_sha256=bindings,probability_bindings=probability_bindings,twelve_records=records,zero_records=zero_records,selection=selected,
        independent_grid_proofs=grid_proofs,source_counts=source_counts,training_budgets=budgets,full_independent_CPU_predictions=replays,slot_order_evidence=slot_proof,matched_replication=rp,
        csv_primary_group_rows=132,csv_descriptive_pool_rows=12,zero_csv_primary_group_rows=22,zero_csv_pool_rows=2,combined_csv_rows=168,
        original_savedGPU_NPZ_and_threshold_unchanged=True,fixed_four_member_ensemble_inherited_via_pinned_actual_audit=True,only_new_max33_eligible=True,
        test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False,reused_development_validation=True,new_independent_confirmation_required=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=report['auditor_source_sha256'],selection=selected,training_budgets=budgets)),flush=True)

def synthetic_slots():
    #2 old docs x64 removed/half =4 new docs x32 selected/half.
    grid=uniform(1024,256);old=[];fresh=[]
    for doc in range(2):
        for role_ix,role in enumerate(ROLES):
            for off in grid:old.append(dict(cache_ix=len(old),sample_ix=2_000_000_000+len(old),split='fit',profile='long-full',source_set='long-full',wrapper='native',document_id=doc,group_id='old:'+str(doc),role=role,prompt_ix=doc*5+role_ix,message_ix=0,token_ix=23+off))
    # Same local prompt IDs in another source set, intentionally lower token_ix.
    old.insert(0,dict(cache_ix=99999,sample_ix=99,split='fit',profile='single',source_set='main',wrapper='native',document_id=90,group_id='nonfull',role='user',prompt_ix=0,message_ix=0,token_ix=0))
    for doc in range(4):
        for role_ix,role in enumerate(ROLES):
            for off in grid:fresh.append(dict(cache_ix=len(fresh),sample_ix=6_000_000_000_000+len(fresh),split='fit',profile='long-full',source_set='long-augmentation',wrapper='native',document_id=6_000_000+doc,group_id='new:'+str(doc),role=role,prompt_ix=doc*5+role_ix,message_ix=0,token_ix=31+off,body_offset=off))
    return pd.DataFrame(old),pd.DataFrame(fresh)


def preflight():
    before=digest(__file__);folder=Path(tempfile.mkdtemp(prefix='long-data-long-mix-audit-preflight-',dir='/tmp'));math=pure_math();old_state=(math.RECORDS.copy(),math.GUARD_CONTROLS.copy(),math.STRICT_CONTROLS.copy(),math.MAX_DIFFERENCE);checks=[]
    def check(name,condition=True):require(bool(condition),'CPU fixture failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except(RuntimeError,ValueError,AssertionError,KeyError):checks.append(name);return
        raise RuntimeError('Expected rejection: '+name)
    import torch
    fit,fresh=synthetic_slots();np.random.seed(21);numpy_rng=np.random.get_state();torch_rng=torch.random.get_rng_state().clone()
    slots,summary=replacement(fit,fresh,old_documents=2,new_documents=4,new_per_half=32)
    check('private_numpy_state_not_consumed',all(np.array_equal(a,b) for a,b in zip(numpy_rng,np.random.get_state())))
    check('torch_rng_not_consumed',torch.equal(torch_rng,torch.random.get_rng_state()))
    check('1280_toy_slots_and_10_equal_rolehalf_buckets',len(slots)==1280 and summary['role_half_buckets']=={r+':'+str(h):128 for r in ROLES for h in [0,1]})
    check('long_profile_filter_precedes_cache_local_offset_derivation',slots.old_body_offset.min()>0 and slots.fit_position.min()>0)
    check('old_kept128_new_selected64_per_doc_role',summary['kept_old_full_rows']==1280 and summary['selected_new_full_rows']==1280 and summary['old_role_rows_per_document']==128 and summary['new_role_rows_per_document']==64)
    again,_=replacement(fit,fresh,old_documents=2,new_documents=4,new_per_half=32);pd.testing.assert_frame_equal(slots,again,check_exact=True);check('fixed_once_scatter_replays_exactly')
    path=folder/'replacement-slots.parquet';slots.to_parquet(path,index=False);pd.testing.assert_frame_equal(pd.read_parquet(path),slots,check_exact=True);check('compact_parquet_exact_mapping_roundtrip')
    bad=fresh.copy();bad.loc[0,'split']='validation';reject('validation_pool_rejected',lambda:replacement(fit,bad,old_documents=2,new_documents=4,new_per_half=32))
    bad=fresh.copy();bad.loc[0,'body_offset']=1;reject('changed_grid_rejected',lambda:replacement(fit,bad,old_documents=2,new_documents=4,new_per_half=32))
    bad=fresh.copy();bad.loc[0,'sample_ix']=fit.sample_ix.iloc[2];reject('sample_collision_rejected',lambda:replacement(fit,bad,old_documents=2,new_documents=4,new_per_half=32))
    bad=fresh.copy();bad.loc[0,'role']='assistant';reject('missing_role_offset_rejected',lambda:replacement(fit,bad,old_documents=2,new_documents=4,new_per_half=32))
    natural=pd.DataFrame(dict(sample_ix=np.arange(4_000_000_000_000,4_000_000_000_006),split='fit',role=['system','user','assistant']*2))
    order=slot_order_evidence(fit,natural,slots,production=False);check('total_slot_label_and_natural_sequence_unchanged',order['five_role_labels_identical_by_slot'] and order['instances']==len(fit)+6)
    bad=slots.copy();bad.loc[0,'new_role']='tool';reject('new_slot_label_change_rejected',lambda:slot_order_evidence(fit,natural,bad,production=False))
    reject('toy_counts_cannot_pass_production_budget',lambda:slot_order_evidence(fit,natural,slots))
    rows,p=math.synthetic();tau,grid,annex=math.direct_grid(rows,p);value=math.evaluate(rows,p,tau)
    check('shared_independent_11group33grid_consumer',len(value['groups'])==11 and grid['groups']==11 and grid['global_argmax_equal'])
    records=[dict(record=n,validation=json.loads(json.dumps(value))) for n in RECORDS];zero=[dict(record=n,validation=json.loads(json.dumps(value))) for n in ZERO_RECORDS]
    result=select_records(math,records,zero,value);check('five_strict_equality_rejections',all(n+':min33_not_strictly_improved' in result['rejection_reasons'] for n in STRICT))
    check('eight_hard_guards_plus_savedGPU',set(result['guards'])==set(GUARDS)|{'original-savedGPU'})
    # Force an otherwise viable abstract metric fixture; independent counting was
    # already exercised above. This isolates added guard/strict control wiring.
    viable=json.loads(json.dumps(value));viable['min33']=.95
    for v in viable['groups'].values():v['system_false_positive_rate']=0.
    peers=json.loads(json.dumps(viable));peers['min33']=.94
    records=[dict(record=n,validation=json.loads(json.dumps(viable if n=='new-max33' else peers))) for n in RECORDS]
    zero=[dict(record=n,validation=json.loads(json.dumps(peers))) for n in ZERO_RECORDS]
    result=select_records(math,records,zero,peers);check('new_only_promotes_when_absolute_strict_and_guards_all_pass',result['promoted'] and result['candidate']=='new-max33')
    diagnostic=json.loads(json.dumps(records));next(r for r in diagnostic if r['record']=='naturalCE-max33')['validation']['min33']=.99
    check('naturalCE_diagnostic_not_hard_strict',select_records(math,diagnostic,zero,peers)['promoted'])
    fail=json.loads(json.dumps(records));next(r for r in fail if r['record']=='priorBinary-max33')['validation']['min33']=.96
    check('priorBinary_additional_strict_is_compulsory',not select_records(math,fail,zero,peers)['promoted'])
    fail=json.loads(json.dumps(records));next(r for r in fail if r['record']=='new-max33')['validation']['min33']=.899999
    check('absolute90_no_fallback',not select_records(math,fail,zero,peers)['promoted'])
    def flatten(rs):
        return pd.DataFrame([dict(record=r['record'],group=g,**{k:v for k,v in m.items() if not isinstance(v,(list,dict))},**{'role_recall_'+role:v for role,v in m['recall'].items()}) for r in rs for g,m in dict(r['validation']['groups'],pooled_descriptive_only=r['validation']['pooled_descriptive_only']).items()])
    main,zframe=flatten(records),flatten(zero);verify_csv(main,records,RECORDS);verify_csv(zframe,zero,ZERO_RECORDS);check('144_main_plus24_zero_168CSV_rows',len(main)==144 and len(zframe)==24)
    main.to_csv(folder/'validation-metrics.csv',index=False);verify_csv(pd.read_csv(folder/'validation-metrics.csv'),records,RECORDS);check('CSV_roundtrip_NA_unsupported_public_CoT_tool')
    reject('missing_prior_record_rejected',lambda:verify_csv(main.iloc[:-1],records,RECORDS))
    weights={'fixture':np.arange(8,dtype=np.float32)};budget=dict(best_epoch=2,best_steps=314,epochs=10,actual_steps=1570,actual_instances=6414930)
    proof=replication_evidence(math,weights,weights,dict(matched=p,priorBinary=p),budget,budget)
    check('matched_repeats_prior_NEW_bitexact_weight_and3bank_p',proof['all_weight_arrays_bitexact'] and proof['all_probability_arrays_bitexact'])
    shifted={b:v.copy() for b,v in p.items()};shifted['old'][0]=np.roll(shifted['old'][0],1)
    reject('matched_fullbank_drift_over_tolerance_rejected',lambda:replication_evidence(math,weights,weights,dict(matched=shifted,priorBinary=p),budget,budget))
    # Pin gate must precede even a status/probability/slots read.
    reject('unreviewed_pins_fail_closed_before_output_access',lambda:require_pins(None,None))
    fake=folder/'terminal';fake.mkdir();(fake/'complete.json').write_text('{}');(fake/'status.json').write_text(json.dumps(dict(stage='training')))
    reject('live_worker_blocks_before_formal_completion_read',lambda:terminal_gate(out=fake,active=[123],source_sha='synthetic',protocol_sha='synthetic'))
    reject('nonterminal_status_rejected',lambda:terminal_gate(out=fake,active=[],source_sha='synthetic',protocol_sha='synthetic'))
    (fake/'status.json').write_text(json.dumps(dict(stage='validation_complete')));terminal_gate(out=fake,active=[],source_sha='synthetic',protocol_sha='synthetic');check('terminal_metadata_gate_accepts_synthetic_only')
    # Independent tiny portable zero-residual functional replay; no real model
    # or vectors, and producer predictor/loader/verifier never imported.
    rng=np.random.RandomState(311);mean=np.zeros(512,np.float32);scale=np.ones(512,np.float32)
    base={'encoder.0.weight':rng.normal(0,.01,(128,512)).astype(np.float32),'encoder.0.bias':rng.normal(0,.01,128).astype(np.float32),
        'head.weight':rng.normal(0,.01,(5,128)).astype(np.float32),'head.bias':np.array([0.,.1,.2,.3,.4],np.float32)}
    state,init=math.initial_binary_state(base);meta=dict(kind='binary-residual',dimensions=512,base_hidden=128,residual_hidden=32,delta_formula='4*tanh(g)',roles=ROLES,
        base_model_sha256=math.CURRENT_SHA,embedded_base_weight_sha256=math.weight_state_sha(base,True),mean_sha256=array_sha(mean),scale_sha256=array_sha(scale))
    model=folder/'synthetic-zero.npz';np.savez(model,mean=mean,scale=scale,metadata=json.dumps(meta),**{'weight::'+k:v for k,v in state.items()})
    meta2,bb,rr=math.binary_model_arrays(model,base,mean,scale);check('independent_portable_base_scaler_zero_last_arrays',all(np.array_equal(a,raw_state(model)[k]) for k,a in state.items()) and init==math.weight_state_sha(state))
    fixture_rows=pd.DataFrame(dict(role=ROLES,cache_ix=np.arange(5)));fixture_bank=rng.normal(0,.1,(5,512)).astype(np.float16)
    computed,original,inv=math.residual_replay(bb,rr,mean,scale,fixture_bank,fixture_rows)
    check('independent_F16_scaled512_base_residual_zero_bitexact_p',np.array_equal(computed.view(np.uint32),original.view(np.uint32)) and inv['delta_min']==inv['delta_max']==inv['delta_mean']==0.)
    t=np.asarray(inv['five_class_transition']);savedraw=dict(tokens=5,raw_non_system_sha256=inv['non_system_raw_after_sha256'],base_raw_non_system_sha256=inv['non_system_raw_before_sha256'],
        raw_non_system_argmax_changes=0,delta_min=0.,delta_max=0.,delta_mean=0.,base_to_residual_p_argmax=t.tolist(),p_conditional_non_system_argmax_changes=inv['p_conditional_non_system_argmax_changes'],
        p_non_system_tie_rows=inv['p_non_system_tie_rows'],p_non_system_underflow_values=inv['p_non_system_underflow_values'],CoT_to_assistant_p_argmax=0,non_system_to_system_p_argmax=0,system_to_non_system_p_argmax=0,non_system_to_different_non_system_p_argmax=0,
        metric_convention='Unmodified savedF32 p.argmax; raw-tail bitidentity does not guarantee p rounding preserves non-system argmax.',
        transition_definition='Rows index original base p.argmax; columns index residual p.argmax, regardless of ground-truth role. CoT_to_assistant is a newly introduced winner transition, not total truth-CoT misclassification.')
    raw_evidence(savedraw,inv,'synthetic/raw');check('rawproof_complete_schema_and_transition_definition')
    reject('rawproof_transition_definition_tamper_rejected',lambda:raw_evidence(savedraw|dict(transition_definition='truth CoT'),inv,'synthetic/raw'))
    badinv=inv.copy();badinv['five_class_transition']=[[0]*5 for _ in range(5)];badinv['five_class_transition'][2][3]=1
    reject('observed_non_system_CoT_assistant_swap_rejected',lambda:raw_evidence(savedraw,badinv,'synthetic/raw'))
    reject('raw_non_system_bit_hash_tamper_rejected',lambda:raw_evidence(savedraw,inv|dict(non_system_raw_after_sha256='bad'),'synthetic/raw'))
    badscale=scale.copy();badscale[0]=2
    reject('portable_scaler_tamper_rejected',lambda:math.binary_model_arrays(model,base,mean,badscale))
    check('old_independent_module_globals_unchanged',old_state==(math.RECORDS,math.GUARD_CONTROLS,math.STRICT_CONTROLS,math.MAX_DIFFERENCE))
    check('production_modules_never_imported',not any(n.startswith('systemness_') for n in __import__('sys').modules))
    check('source_immutable_during_preflight',before==digest(__file__))
    report=dict(stage='synthetic_long_mix_result_auditor_complete_adapter',source_sha256=before,pure_math_source_path=str(MATH_SOURCE),pure_math_source_sha256=MATH_SHA,checks=checks,check_count=len(checks),
        trainer_source_sha256=SOURCE_SHA,trainer_protocol_sha256=PROTOCOL_SHA,actual_enabled=True,actual_executed=False,final_completion_adapter_frozen=True,synthetic_only=True,
        real_replacement_slots_read=False,new_probabilities_read=False,test_data_or_results_read=False,gpu_used=False,formal_outputs_written=False)
    output=folder/'report.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(report=str(output),report_sha256=digest(output),source_sha256=before,checks=len(checks))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--preflight',action='store_true');group.add_argument('--actual',action='store_true');parser.add_argument('--output',default='/tmp/long-data-long-mix-residual-actual-audit.json');args=parser.parse_args()
    if args.preflight:preflight()
    elif args.actual:actual(args.output)
    else:parser.print_help()

if __name__=='__main__':main()
