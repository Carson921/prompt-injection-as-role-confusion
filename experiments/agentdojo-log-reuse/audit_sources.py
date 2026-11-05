"""Check source payload/spec bindings, independently of outcome selection."""
import shutil
from reuse_common import *


def main():
    manifest=read(ROOT/'manifest.json')
    assert manifest['plan_sha256']==sha(ROOT/'plan.json')
    schedule={r['case_id']:r for r in read(BASE/'schedule.json')}
    native=read(NATIVE_OLD/'benchmark_manifest.json')
    assert native['benchmark_version']=='v1.2.1'
    assert native['payloads']['chatinject']==schedule[next(iter(schedule))]['payload_template']
    assert native['run_identity']['model_revision']==read(BASE/'manifest.json')['model_binding']['revision']
    copied={}
    for origin,names in ((NATIVE_OLD,('benchmark_manifest.json','run_identity.json')),
                         (BASE,('manifest.json','schedule.json','agentdojo-source.json')),
                         (OLD_ASR,('manifest.json','validation.json','snapshot_validation.json')),
                         (PREVIOUS_CONTINUATION,('manifest.json','validation.json','snapshot_validation.json'))):
        for name in names:
            target=ROOT/'source_bindings'/origin.name/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(origin/name,target)
            assert sha(target)==sha(origin/name)
            copied[str(target.relative_to(ROOT))]=sha(target)
    counts={}
    for r in read(ROOT/'plan.json'):
        case=schedule[r['case_id']]
        if r['asr_source']=='historical_native_complete':
            record=read(ROOT/r['sources'][0]['path'])
            assert record['spec_sha256']==native['spec_sha256']
            assert record['condition']=='chatinject'
            assert all(record[k]==case[k] for k in ('suite','user_task_id','injection_task_id'))
        elif r['asr_source']=='historical_reference':
            refid=r['case_id'].replace('plain_delimiter/','chatinject/',1).replace('remember_delimiter/','prerequisite_chatinject/',1)
            record=next(x for x in read(REFERENCE/'cases.json') if x['case_id']==refid)
            assert record['payload_template']==case['payload_template']
            assert record['actual_goal']==case['actual_goal']
        elif r['asr_source']=='historical_table1':
            record=read(ROOT/r['sources'][0]['path'])
            assert record['benchmark_version']=='v1.2.1'
            assert record['injections']==case['injections']
            assert record['error'] is None
        elif r['asr_source']=='completed_previous_continuation':
            record=read(Path(r['native_result']))
            assert sha(Path(r['native_result']))==r['native_result_sha256']
            assert record['status']=='completed' and record['case_id']==r['case_id']
            assert record['attacker_success']==r['attacker_success'] and record['utility']==r['utility']
        counts[r['asr_source']]=counts.get(r['asr_source'],0)+1
    write(ROOT/'provenance_audit.json',{'status':'passed','plan_sha256':sha(ROOT/'plan.json'),
          'historical_payloads_and_task_ids_verified':True,'native_spec_bindings_verified':True,
          'counts':counts,'source_bindings_sha256':copied})
    print(serial({'status':'passed','counts':counts}))


if __name__=='__main__':main()
