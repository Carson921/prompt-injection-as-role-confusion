"""Bounded CPU diagnostic: old/new FIT feature values only, no predictions.

--run explicitly reads deterministic uniform sampled fit rows. --preflight is
synthetic. No old global verifier, LM, validation/test feature or p is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
sys.path.insert(0,str(HERE))
import systemness_head_probes as hp

OLD=Path('/dev/shm/role-analysis-top8-heads')
NEW=Path('/dev/shm/role-analysis-long-augmentation-heads')
SCALER=hp.OUT/'standardization.npz'
OUTPUT=Path('/tmp/long_data_fit_feature_distribution_diagnostic.json')
CAP=4096
ROLES=['system','user','cot','assistant','tool']


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def arrsha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require(value,message):
    if not value:raise RuntimeError(message)


def rounded(value):
    if np.isscalar(value):return float(np.round(value,6))
    return [rounded(x) for x in np.asarray(value)]


def sample_rows(frame,cap=CAP):
    require(frame.split.eq('fit').all() and len(frame)>0,'Feature reads must be nonempty fit-only.')
    frame=frame.sort_values('cache_ix')
    select=np.linspace(0,len(frame)-1,min(cap,len(frame)),dtype=np.int64)
    chosen=frame.iloc[select]
    require(chosen.cache_ix.is_unique,'Repeated sampled row.')
    return chosen


def stats(raw,mean,scale):
    require(raw.ndim==2 and raw.shape[1]==512 and np.isfinite(raw).all(),'Invalid sampled fit features.')
    z=((raw.astype(np.float32)-mean)/scale).reshape(-1,8,64)
    r=raw.astype(np.float32).reshape(-1,8,64)
    absolute=np.abs(z);raw_absolute=np.abs(r)
    token_norm=np.linalg.norm(z,axis=2);raw_norm=np.linalg.norm(r,axis=2)
    result=dict(sampled_rows=len(raw),nonfinite_values=0,
        raw_element_rms=rounded(np.sqrt(np.mean(r.astype(np.float64)**2,axis=(0,2)))),
        raw_abs_p99=rounded(np.quantile(raw_absolute,.99,axis=(0,2))),
        raw_abs_max=rounded(raw_absolute.max(axis=(0,2))),
        raw_token_l2_p50=rounded(np.median(raw_norm,axis=0)),
        raw_token_l2_p99=rounded(np.quantile(raw_norm,.99,axis=0)),
        z_element_mean=rounded(np.mean(z,axis=(0,2),dtype=np.float64)),
        z_element_std=rounded(np.std(z.astype(np.float64),axis=(0,2))),
        z_element_rms=rounded(np.sqrt(np.mean(z.astype(np.float64)**2,axis=(0,2)))),
        z_abs_p99=rounded(np.quantile(absolute,.99,axis=(0,2))),
        z_abs_max=rounded(absolute.max(axis=(0,2))),
        z_token_l2_p50=rounded(np.median(token_norm,axis=0)),
        z_token_l2_p99=rounded(np.quantile(token_norm,.99,axis=0)),
        z_token_l2_max=rounded(token_norm.max(axis=0)),
        z_abs_gt3_fraction=rounded(np.mean(absolute>3,axis=(0,2))),
        z_abs_gt5_fraction=rounded(np.mean(absolute>5,axis=(0,2))),
        z_abs_gt10_fraction=rounded(np.mean(absolute>10,axis=(0,2))),
        z_abs_gt20_fraction=rounded(np.mean(absolute>20,axis=(0,2))))
    return result,dict(mean=z.mean(0,dtype=np.float64),std=z.std(0,dtype=np.float64),
        minimum=z.min(0),maximum=z.max(0),sample=z)


def compare(new,old,summary_new,summary_old):
    def ratio(a,b):
        return [None if y==0 else round(float(x/y),6) for x,y in zip(a,b)]
    sample=new['sample']
    outside=(sample<old['minimum'][None,:,:])|(sample>old['maximum'][None,:,:])
    location=np.sqrt(np.mean((new['mean']-old['mean'])**2,axis=1))
    variance=np.sqrt(np.mean(new['std']**2,axis=1))/np.maximum(np.sqrt(np.mean(old['std']**2,axis=1)),1e-12)
    return dict(z_element_rms_ratio=ratio(summary_new['z_element_rms'],summary_old['z_element_rms']),
        z_token_l2_p99_ratio=ratio(summary_new['z_token_l2_p99'],summary_old['z_token_l2_p99']),
        z_abs_p99_ratio=ratio(summary_new['z_abs_p99'],summary_old['z_abs_p99']),
        channel_mean_shift_rms_old_global_sigma_units=rounded(location),
        channel_within_std_rms_ratio=rounded(variance),
        fraction_new_elements_outside_old_sample_coordinate_minmax=rounded(outside.mean(axis=(0,2))),
        fraction_new_tokens_any_coordinate_outside_old_sample_minmax=rounded(outside.any(axis=2).mean(axis=0)))


def run():
    source_sha=sha(__file__)
    rows_old=pd.read_parquet(OLD/'rows.parquet');rows_new=pd.read_parquet(NEW/'rows.parquet')
    fit,valid,partition=hp.partitions(rows_old)  # Row descriptors only; no bank/p access.
    old_raw_fit=rows_old[rows_old.split.eq('fit')]
    removed=old_raw_fit[~old_raw_fit.cache_ix.isin(fit.cache_ix)]
    require(len(fit)==578090 and len(removed)==4160 and removed.document_id.nunique()==26,
        'Original densefit replacement changed.')
    full=fit[fit.profile.eq('long-full')&fit.source_set.eq('long-full')]
    fresh=rows_new[rows_new.split.eq('fit')].copy()
    require(len(full)==168960 and full.document_id.nunique()==132 and len(fresh)==327680
        and fresh.document_id.nunique()==256 and fresh.profile.eq('long-full').all()
        and fresh.source_set.eq('long-augmentation').all(),'True1024 fit pool/newfit layout changed.')
    require(not set(fit.cache_ix)&set(valid.cache_ix) and set(fit.role)==set(fresh.role)==set(ROLES),'Split/role leak.')
    # Base descriptors predate body_offset. Derive message-local ordinal solely
    # from token positions, as the frozen trainer does; no source/token decode.
    minimum=fit.groupby(['source_set','prompt_ix','message_ix']).token_ix.transform('min')
    fit=fit.copy();fit['diagnostic_offset']=fit.token_ix-minimum
    full=fit[fit.profile.eq('long-full')&fit.source_set.eq('long-full')]
    fresh['diagnostic_offset']=fresh.body_offset
    expected_grid=np.linspace(0,1023,256,dtype=np.int64)
    for label,frame in [('old',full),('new',fresh)]:
        for _,part in frame.groupby(['document_id','role'],sort=False):
            require(np.array_equal(part.sort_values('token_ix').diagnostic_offset.to_numpy(),expected_grid),
                label+' truefull fit token grid differs from frozen256 sampling.')
    with np.load(SCALER,allow_pickle=False) as archive:mean=archive['mean'].copy();scale=archive['scale'].copy()
    require(mean.shape==scale.shape==(512,) and mean.dtype==scale.dtype==np.float32
        and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale>0).all(),'Frozen scaler changed.')
    evidence_old=json.loads((hp.FEATURE_OUT/'extraction-complete.json').read_text())
    evidence_new=json.loads((hp.data.runtime.OUT/'systemness-long-augmentation/extraction-complete.json').read_text())
    heads=evidence_old['selected_head_ids']
    require(heads==evidence_new['selected_head_ids'] and len(heads)==8,'Head rank order mismatch.')
    banks={}
    for name,folder,rows in [('old',OLD,rows_old),('new',NEW,rows_new)]:
        require((folder/'features.f16').stat().st_size==len(rows)*512*2,'Bank byte count changed.')
        banks[name]=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(len(rows),512))
    domains={'old-allfit':('old',fit),'old-true1024-fit':('old',full),'new-true1024-fit':('new',fresh)}
    forbidden={'old':set(rows_old[rows_old.split.ne('fit')].cache_ix),
        'new':set(rows_new[rows_new.split.ne('fit')].cache_ix)}
    summaries={name:{} for name in domains};comparisons={};read_rows={'old':set(),'new':set()};selection={}
    # At most three 4096x512 sample matrices are live at once (~24MiB F32),
    # plus bounded headwise temporary arrays. No entire bank is materialized.
    for role in ROLES:
        for half in ['first512','tail512plus']:
            key=role+'/'+half;temporary={}
            for name,(bank,frame) in domains.items():
                subset=frame[frame.role.eq(role)&(frame.diagnostic_offset.lt(512) if half=='first512' else frame.diagnostic_offset.ge(512))]
                chosen=sample_rows(subset);indices=chosen.cache_ix.to_numpy(dtype=np.int64)
                require(chosen.split.eq('fit').all() and not set(indices)&forbidden[bank],
                    'Attempted feature read outside fit.')
                raw=np.asarray(banks[bank][indices],dtype=np.float32)
                read_rows[bank].update(map(int,indices));result,detail=stats(raw,mean,scale)
                result.update(available_fit_rows=len(subset),available_documents=int(subset.document_id.nunique()),
                    sampled_documents=int(chosen.document_id.nunique()),sample_cache_ix_sha256=arrsha(indices),
                    sample_sample_ix_sha256=arrsha(chosen.sample_ix.to_numpy(dtype=np.int64)))
                summaries[name][key]=result;temporary[name]=detail
                selection[name+'/'+key]=dict(rows=len(indices),cache_ix_sha256=arrsha(indices),sample_ix_sha256=result['sample_sample_ix_sha256'])
            comparisons[key]={reference:compare(temporary['new-true1024-fit'],temporary[reference],
                summaries['new-true1024-fit'][key],summaries[reference][key]) for reference in ['old-allfit','old-true1024-fit']}
            del temporary,raw
    removed_counts=removed.groupby(['profile','role']).size().unstack(fill_value=0).to_dict()
    report=dict(stage='fit_only_bounded_feature_distribution_diagnostic',source_sha256=source_sha,
        scope=dict(fit_feature_values_read_only=True,validation_feature_values_read=False,test_data_read=False,
            probabilities_or_predictions_read=False,validation_metrics_read=False,model_forward_run=False,gpu_used=False,
            row_descriptors_include_validation_for_split_guards_only=True),
        sample=dict(policy='Uniform integer linspace over ascending cache_ix, independently by domain/role/message-body-half; at most4096 rows per stratum.',
            cap_per_role_half_domain=CAP,selection_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),
            distinct_old_fit_feature_rows_read=len(read_rows['old']),distinct_new_fit_feature_rows_read=len(read_rows['new'])),
        identity=dict(old_rows_sha256=sha(OLD/'rows.parquet'),new_rows_sha256=sha(NEW/'rows.parquet'),
            scaler_sha256=sha(SCALER),scaler_mean_sha256=arrsha(mean),scaler_scale_sha256=arrsha(scale),
            scaler_scale_min=float(scale.min()),scaler_floor_coordinates=int((scale<=1e-6).sum()),
            old_feature_sha256_audited_declaration=evidence_old['features_sha256'],
            new_feature_sha256_audited_declaration=evidence_new['features_sha256'],raw_bank_full_hash_reread=False,
            hp_partitions_source_sha256=sha(hp.__file__),old_fit_cache_ix_sha256=partition['fit_cache_ix_sha256']),
        heads_in_feature_column_order=[dict(rank=i,layer=int(h//64),head=int(h%64)) for i,h in enumerate(heads)],
        fit_layout=dict(old_rows=len(fit),new_rows=len(fresh),old_true1024_rows=len(full),old_true1024_documents=132,
            new_true1024_documents=256,old_profile_row_counts=fit.profile.value_counts().to_dict(),
            offsets='Old bank has no explicit body_offset; prompt_ix is source-cache-local, so derive token_ix minus first token per(source_set,prompt_ix,message_ix). Independently verify each old/new true1024 document/role has identical256-point uniform grid0..1023, as the frozen trainer packets helper does per document/role.',
            old_profile_role_rows=fit.groupby(['profile','role']).size().unstack(fill_value=0).to_dict(),
            removed_sparse_rows=len(removed),removed_sparse_documents=int(removed.document_id.nunique()),
            removed_sparse_profile_roles=removed_counts,
            overlap_policy='Densefit removes exactly26 existing sparse long-single document packets (4160=26*5*32 rows) overlapping132 true1024 fit sources. Remaining106 full documents had no old long-single packet. Do not remove132 full packets or replace them again.'),
        summaries=summaries,comparisons=comparisons,
        interpretation_limits=['Descriptive sampled fit distributions, not p-values or F1 estimates. Token positions/roles from a source are paired and correlated.',
            'Old-allfit versus newfit mixes profile/length composition; old-true1024 comparison controls profile, token grid and halves but not document identity.',
            'Coordinate min/max exceedance uses sampled empirical support and is not proof of population support violation or extraction error.',
            'No causal claim about negative development results, no new source selection/reranking, no training or calibration change.'])
    require(sha(__file__)==source_sha,'Diagnostic source changed while running.')
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(report_path=str(OUTPUT),report_sha256=sha(OUTPUT),source_sha256=source_sha,
        distinct_fit_feature_rows_read=report['sample'],heads=report['heads_in_feature_column_order'])),flush=True)


def preflight():
    rng=np.random.RandomState(3);raw=rng.normal(size=(32,512)).astype(np.float32)
    summary,detail=stats(raw,np.zeros(512,np.float32),np.ones(512,np.float32))
    comparison=compare(detail,detail,summary,summary)
    require(comparison['z_element_rms_ratio']==[1.]*8 and comparison['fraction_new_elements_outside_old_sample_coordinate_minmax']==[0.]*8,
        'Identity scale/support fixture failed.')
    shifted,shifted_detail=stats(raw*3+7,np.zeros(512,np.float32),np.ones(512,np.float32))
    require(min(compare(shifted_detail,detail,shifted,summary)['z_element_rms_ratio'])>2,'Scale-shift fixture failed.')
    print(json.dumps(dict(synthetic_cpu_pass=True,real_features_read=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true');action.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight()
    elif args.run:run()
    else:parser.print_help()
