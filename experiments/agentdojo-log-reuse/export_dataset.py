"""Export one portable probe dataset, retaining outcome and episode provenance."""
import argparse
from collections import Counter
import csv
import fcntl
import io
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from reuse_common import *


def copy_verified(source, target, digest=None):
    source,target=Path(source),Path(target)
    digest=digest or sha(source)
    assert sha(source)==digest
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copyfile(source,target)
    assert sha(target)==digest
    return {'path':str(target.relative_to(ROOT)),'sha256':digest}


def summary(records):
    completed=[r for r in records if r['status']=='completed']
    return {'cases':len(records),'completed':len(completed),
            'attacker_successes':sum(r['attacker_success'] for r in completed),
            'asr':sum(r['attacker_success'] for r in completed)/len(completed) if completed else None,
            'utility_successes':sum(r['utility'] for r in completed),
            'asr_denominator':'completed cases only'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    lock=(ROOT/'export.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    manifest=read(ROOT/'manifest.json')
    assert manifest['plan_sha256']==sha(ROOT/'plan.json')
    assert manifest['schedule_sha256']==sha(BASE/'schedule.json')
    provenance=read(ROOT/'provenance_audit.json')
    assert provenance['status']=='passed' and provenance['plan_sha256']==manifest['plan_sha256']
    previous_audit=read(ROOT/'previous_completed_audit.json')
    assert previous_audit['status']=='partial_verified'
    assert previous_audit['cases']==manifest['previous_completed_native_cases']
    assert read(ROOT/'feature_terminal.json')['status']=='complete'
    old_audit=[json.loads(line) for line in (ROOT/'current67_audit.log').read_text().splitlines() if line.startswith('{')][-1]
    assert old_audit['status']=='partial_verified' and old_audit['cases']==67
    if not args.partial:
        audit=read(MISSING/'audit.json')
        assert audit['status']=='passed' and audit['cases']==manifest['missing_asr_cases']
    plan=read(ROOT/'plan.json')
    canonical={r['case_id']:r for r in read(BASE/'canonical_cases.json')}
    feature_index={r['case_id']:r for r in read(BASE/'feature_index.json')}
    protocol=read(BASE/'manifest.json')
    slots=[protocol['head_ids_union'].index(h) for h in protocol['head_ids_final_v7_16']]
    records=[]
    rows=0
    state_paths={}
    for planned in plan:
        r=dict(planned)
        cid=r['case_id']
        r.update(status='completed' if type(r['attacker_success']) is bool else 'pending_asr',row_start=None,row_end=None)
        if r['asr_source']=='needs_native_continuation':
            path=MISSING/'cases'/(cid+'.json')
            if path.exists():
                result=read(path)
                assert result['status']=='completed',result
                r.update(status='completed',attacker_success=result['attacker_success'],utility=result['utility'],asr_source='new_missing_native_continuation',
                         native_result=str(path.relative_to(ROOT)),native_result_sha256=sha(path))
        if not args.partial:
            assert r['status']=='completed',cid
        for source in r['sources']:
            assert sha(ROOT/source['path'])==source['sha256']
        if r['payload_exposed']:
            if r['feature_method']=='forward_logged_input':
                input_path=ROOT/r['input_path']
                assert sha(input_path)==r['input_sha256']
                receipt=read(input_path.with_name('capture.json'))
                assert receipt['status']=='captured' and receipt['input_sha256']==sha(input_path)
                source=input_path.with_name('states.safetensors')
                digest=receipt['states_sha256']
            else:
                source=Path(r['states'])
                digest=r['states_sha256']
            local=ROOT/'states'/(cid+'.safetensors')
            receipt=copy_verified(source,local,digest)
            state_paths[cid]=local
            r.update(states=receipt['path'],states_sha256=receipt['sha256'],row_start=rows,row_end=rows+r['rows'])
            rows+=r['rows']
            # Canonical states also carry their full original model input.
            if 'input_path' not in r:
                origin=canonical[cid]
                call=phase(origin)/'calls'/origin['first_exposure_call']
                for name in ('input.json','original_request.json'):
                    copied=copy_verified(call/name,ROOT/'logged_inputs'/cid/name)
                    if name=='input.json':
                        r.update(input_path=copied['path'],input_sha256=copied['sha256'])
        if r['asr_source'] in ('current_completed_native','completed_previous_continuation'):
            result=read(Path(r['native_result']))
            native_source=OLD_ASR if r['asr_source']=='current_completed_native' else PREVIOUS_CONTINUATION
            for item in result['requests']:
                for name in ('original_request.json','input.json','response.json','parsed_response.json'):
                    copy_verified(native_source/'calls'/item['call_id']/name,ROOT/'sources'/cid/'native_calls'/f"call_{item['call_index']:04d}"/name)
            for i,invocation in enumerate(result['trajectory_invocations']):
                copy_verified(native_source/invocation['path'],ROOT/'sources'/cid/f'trajectory_{i:02d}.json',invocation['sha256'])
            r['native_result']=str((ROOT/'sources'/cid/'case.json').relative_to(ROOT))
        records.append(r)
    assert len(records)==manifest['total_cases']==778
    assert set(r['suite'] for r in records)=={'banking','slack','travel'}
    assert set(r['condition'] for r in records)=={'plain_delimiter','remember_delimiter'}
    out=ROOT/'probe_inputs'
    out.mkdir(exist_ok=True)
    feature_binding=signature({'plan':manifest['plan_sha256'],'rows':rows,
                  'states':[(r['case_id'],r.get('states_sha256')) for r in records]})
    receipt_path=out/'features.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        assert receipt['binding']==feature_binding
        assert all(sha(out/name)==digest for name,digest in receipt['sha256'].items())
    else:
        arrays={'X_final_v7_16':np.lib.format.open_memmap(out/'X_final_v7_16.npy',mode='w+',dtype=np.float32,shape=(rows,1024)),
                'case_index':np.lib.format.open_memmap(out/'case_index.npy',mode='w+',dtype=np.int32,shape=(rows,)),
                'token_ids':np.lib.format.open_memmap(out/'token_ids.npy',mode='w+',dtype=np.int64,shape=(rows,)),
                'positions':np.lib.format.open_memmap(out/'positions.npy',mode='w+',dtype=np.int64,shape=(rows,))}
        for name in ('payload_mask','goal_mask','template_mask','context_mask'):
            arrays[name]=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=np.bool_,shape=(rows,))
        for r in records:
            if not r['payload_exposed']:
                continue
            state=load_file(str(state_paths[r['case_id']]),device='cpu')
            value=state['features_union'][:,slots,:].float().numpy().reshape(-1,1024)
            assert value.shape==(r['rows'],1024) and np.isfinite(value).all()
            assert not np.isin(state['token_ids'].numpy(),protocol['delimiter_ids']).any()
            assert not (state['goal_mask'] & ~state['payload_mask']).any()
            a,b=r['row_start'],r['row_end']
            arrays['X_final_v7_16'][a:b]=value
            arrays['case_index'][a:b]=r['case_index']
            for name in arrays:
                if name not in ('X_final_v7_16','case_index'):
                    arrays[name][a:b]=state[name].numpy()
        for array in arrays.values():
            array.flush()
        del arrays
        hashes={name:sha(out/name) for name in ('X_final_v7_16.npy','case_index.npy','token_ids.npy','positions.npy',
                   'payload_mask.npy','goal_mask.npy','template_mask.npy','context_mask.npy')}
        # Verify actual exported feature values, masks and indices in a second read.
        matrix=np.load(out/'X_final_v7_16.npy',mmap_mode='r')
        indices=np.load(out/'case_index.npy',mmap_mode='r')
        for r in records:
            if r['payload_exposed']:
                state=load_file(str(state_paths[r['case_id']]),device='cpu')
                a,b=r['row_start'],r['row_end']
                assert np.array_equal(matrix[a:b],state['features_union'][:,slots,:].float().numpy().reshape(-1,1024))
                assert np.all(indices[a:b]==r['case_index'])
                for name in ('token_ids','positions','payload_mask','goal_mask','template_mask','context_mask'):
                    assert np.array_equal(np.load(out/(name+'.npy'),mmap_mode='r')[a:b],state[name].numpy())
        write(receipt_path,{'status':'passed','binding':feature_binding,'rows':rows,'shape':[rows,1024],
               'head_ids':protocol['head_ids_final_v7_16'],'all_exported_values_readback_verified':True,'sha256':hashes})
    labels=np.asarray([int(r['attacker_success']) if r['status']=='completed' else -1 for r in records],dtype=np.int8)
    utility=np.asarray([int(r['utility']) if r['status']=='completed' else -1 for r in records],dtype=np.int8)
    indices=np.load(out/'case_index.npy',mmap_mode='r')
    for name,value in {'attacker_success_by_case':labels,'attacker_success_by_token':labels[indices],
                      'utility_by_case':utility,'utility_by_token':utility[indices],
                      'asr_available_by_case':labels>=0,'asr_available_by_token':labels[indices]>=0}.items():
        np.save(out/(name+'.npy'),value,allow_pickle=False)
        assert np.array_equal(np.load(out/(name+'.npy'),mmap_mode='r'),value)
    for r in records:
        r['asr_label']=int(r['attacker_success']) if r['status']=='completed' else None
    report={'status':'complete' if np.all(labels>=0) else 'partial', 'all_cases':summary(records),
            'payload_exposed':summary([r for r in records if r['payload_exposed']]),
            'unexposed':summary([r for r in records if not r['payload_exposed']]),
            'asr_source_counts':dict(Counter(r['asr_source'] for r in records)),
            'feature_method_counts':dict(Counter(str(r['feature_method']) for r in records)),
            'rows':rows,'shape':[rows,1024],'system_order':'original normal',
            'by_condition':{c:summary([r for r in records if r['condition']==c]) for c in ('plain_delimiter','remember_delimiter')},
            'by_condition_and_suite':[{'condition':c,'suite':s,**summary([r for r in records if r['condition']==c and r['suite']==s])}
                for c in ('plain_delimiter','remember_delimiter') for s in manifest['suites']],
            'by_user_task':[{'condition':c,'suite':s,'user_task_id':u,**summary([r for r in records if (r['condition'],r['suite'],r['user_task_id'])==(c,s,u)])}
                for c,s,u in sorted({(r['condition'],r['suite'],r['user_task_id']) for r in records})]}
    write(ROOT/'case_results.json',records)
    write(out/'cases.json',records)
    write(ROOT/'asr_summary.json',report)
    fields=['case_index','case_id','condition','suite','user_task_id','injection_task_id','status','attacker_success','utility',
            'asr_label','asr_source','payload_exposed','feature_method','row_start','row_end','states']
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    (ROOT/'case_results.csv').write_text(stream.getvalue())
    (out/'USAGE.txt').write_text('''X_final_v7_16.npy: float32[N,1024], unstandardized selected 16 x 64 heads.
Every row is a non-delimiter token in the first complete-payload trusted tool response.
case_index.npy maps token rows to cases.json (all 778 cases, including unexposed).
attacker_success_by_case.npy and attacker_success_by_token.npy: int8, 0/1 success.
Only partial exports may contain -1 for pending ASR; filter asr_available_* masks.
Unexposed episodes have no feature rows and remain in the case-level ASR denominator.
Masks identify the payload, goal, template and benign context within each tool body.
System message order is original/normal. Historical logs retain their original outcomes.
Historical inputs with saved prompt and IDs are exact. Table 1 dialogue inputs are
reconstructed using v1.2.1 tools, evaluation date and medium reasoning, explicitly marked.
Historical reference exposure-conditioned selections are retained in case metadata.
Mixed backend/selection provenance is explicit; this is a merged existing-results cohort.
''')
    write(ROOT/'dataset_audit.json',{'status':'passed','full_benchmark_complete':report['status']=='complete',
          'cases':len(records),'completed_asr':int((labels>=0).sum()),'rows':rows,
          'features_readback_verified':True,'token_labels_aligned':True,'no_delimiter_rows':True,
          'plan_sha256':sha(ROOT/'plan.json'),'features_receipt_sha256':sha(receipt_path),
          'source_selection_uses_outcome':False,'case_results_sha256':sha(ROOT/'case_results.json')})
    print(serial(report['all_cases']),flush=True)


if __name__=='__main__':
    import json
    main()
