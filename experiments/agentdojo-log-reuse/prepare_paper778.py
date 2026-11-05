"""Restrict the existing log-first merge to the confirmed 778 delimiter cases."""
from collections import Counter
import shutil

from reuse_common import *


def copy_bound(source,target,digest=None):
    digest=digest or sha(source)
    assert sha(source)==digest
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:target.hardlink_to(source)
        except OSError:shutil.copyfile(source,target)
    assert sha(target)==digest


def main():
    assert not (ROOT/'plan.json').exists(),'A paper778 plan is already frozen'
    ROOT.mkdir(parents=True,exist_ok=True)
    evidence=LEGACY_MERGED/'scope_correction/completed_native_audit.log'
    audited=[json.loads(line) for line in evidence.read_text().splitlines() if line.startswith('{')][-1]
    assert audited['status']=='partial_verified'
    plan=[];imported=[]
    for item in read(LEGACY_MERGED/'plan.json'):
        if item['suite']=='workspace':continue
        r=dict(item);cid=r['case_id']
        r['case_index']=len(plan)
        assert r['condition'] in ('plain_delimiter','remember_delimiter')
        for source in r['sources']:
            copy_bound(LEGACY_MERGED/source['path'],ROOT/source['path'],source['sha256'])
        if r.get('input_path'):
            old=LEGACY_MERGED/r['input_path']
            copy_bound(old,ROOT/r['input_path'],r['input_sha256'])
            for path in old.parent.iterdir():
                if path.is_file() and path.name!='input.json':
                    copy_bound(path,ROOT/r['input_path'].replace('input.json',path.name))
        path=PREVIOUS_CONTINUATION/'cases'/(cid+'.json')
        if r['asr_source']=='needs_native_continuation' and path.exists():
            result=read(path)
            assert result['status']=='completed' and type(result['attacker_success']) is bool
            r.update(asr_source='completed_previous_continuation',attacker_success=result['attacker_success'],
                     utility=result['utility'],native_result=str(path),native_result_sha256=sha(path))
            target=ROOT/'sources'/cid/'case.json'
            copy_bound(path,target)
            r['sources'].append({'original_path':str(path),'path':str(target.relative_to(ROOT)),'sha256':sha(target)})
            imported.append({'case_id':cid,'result_sha256':sha(path)})
        plan.append(r)
    assert len(plan)==778 and len({r['case_id'] for r in plan})==778
    assert len(imported)==audited['cases']
    assert Counter(r['condition'] for r in plan)=={'plain_delimiter':389,'remember_delimiter':389}
    assert Counter(r['suite'] for r in plan)=={'banking':288,'slack':210,'travel':280}
    missing=[r['case_id'] for r in plan if r['asr_source']=='needs_native_continuation']
    assert all(r.startswith('remember_delimiter/') for r in missing),'All Plain outcomes already exist'
    counts=Counter(r['asr_source'] for r in plan)
    write(ROOT/'plan.json',plan)
    write(ROOT/'missing_case_ids.json',missing)
    protocol=read(BASE/'manifest.json')
    write(ROOT/'manifest.json',{'format':'agentdojo-paper778-log-first-merge-v2','total_cases':778,
          'suites':['banking','slack','travel'],'conditions':['plain_delimiter','remember_delimiter'],
          'user_tasks':57,'task_pairs':389,'delimiter_only':True,'system_order':'original normal',
          'schedule_sha256':sha(BASE/'schedule.json'),'base_manifest_sha256':sha(BASE/'manifest.json'),
          'plan_sha256':sha(ROOT/'plan.json'),'missing_case_ids_sha256':sha(ROOT/'missing_case_ids.json'),
          'counts':dict(counts),'missing_asr_cases':len(missing),'previous_completed_native_cases':len(imported),
          'reused_asr_cases':778-len(missing),'previous_collection':str(LEGACY_MERGED),
          'feature_heads':protocol['head_ids_final_v7_16'],'feature_head_ids_union':protocol['head_ids_union'],
          'delimiter_ids':protocol['delimiter_ids'],'selection_uses_outcome':False,
          'source_code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
    write(ROOT/'previous_completed_audit.json',{**audited,'source_log_sha256':sha(evidence),
          'previous_manifest_sha256':sha(PREVIOUS_CONTINUATION/'manifest.json'),'cases_imported':imported})
    for source,name in [(evidence,'previous_completed_audit.log'),(LEGACY_MERGED/'current67_audit.log','current67_audit.log'),
                        (LEGACY_MERGED/'feature_terminal.json','feature_terminal.json'),(LEGACY_MERGED/'feature_progress.json','feature_progress.json')]:
        copy_bound(source,ROOT/name)
    write(ROOT/'progress.json',{'stage':'prepared_paper778','total':778,'reused_asr':778-len(missing),
          'new_asr_total':len(missing),'new_asr_completed':0,'asr_completed':778-len(missing),
          'logged_feature_forwards_completed':266,'logged_feature_forwards_total':266})
    print(serial({'status':'prepared','cases':778,'reused_asr':778-len(missing),'missing_asr':len(missing),
                  'sources':dict(counts),'feature_rows':sum(r.get('rows',0) for r in plan)}))


if __name__=='__main__':main()
