"""Freeze a filtered native continuation using the already validated engine."""
from pathlib import Path
import shutil

from reuse_common import *


def change(text, before, after):
    assert text.count(before)==1, before
    return text.replace(before,after)


def main():
    plan=read(ROOT/'plan.json')
    missing=read(ROOT/'missing_case_ids.json')
    assert {r['case_id'] for r in plan if r['asr_source']=='needs_native_continuation'}==set(missing)
    runtime=MISSING/'provenance'
    runtime.mkdir(parents=True,exist_ok=True)
    source=OLD_ASR/'provenance'
    common=(source/'common.py').read_text()
    common=change(common,"OUTPUT = BASE.with_name(BASE.name + '_asr')", f"MERGED = Path({str(ROOT)!r})\nOUTPUT = MERGED / 'missing_asr'")
    (runtime/'common.py').write_text(common)
    bridge=(source/'continuation_bridge.py').read_text()
    bridge=change(bridge,"schedule = read(BASE / 'schedule.json')", "selected = set(read(output / 'selected_case_ids.json'))\n    schedule = [r for r in read(BASE / 'schedule.json') if r['case_id'] in selected]\n    assert len(schedule) == len(selected)")
    (runtime/'continuation_bridge.py').write_text(bridge)
    runner=(source/'run.py').read_text()
    runner=change(runner,"'format': 'agentdojo-native-asr-continuation-v1'", "'format': 'agentdojo-filtered-native-asr-continuation-v2', 'selected_case_ids_sha256': sha(OUTPUT / 'selected_case_ids.json')")
    runner=change(runner,"canonical = {r['case_id']: r for r in read(BASE / 'canonical_cases.json')}", "selected = set(read(OUTPUT / 'selected_case_ids.json'))\n    canonical = {r['case_id']: r for r in read(BASE / 'canonical_cases.json') if r['case_id'] in selected}\n    assert len(canonical) == len(selected)")
    runner=change(runner,"engine.validate(model, tok, original['head_ids_union'], OUTPUT, original['query_tile'])", "engine.validate(model, tok, original['head_ids_union'], OUTPUT, original['query_tile'])\n    from logged_features import capture_logged\n    capture_logged(model, tok, engine)")
    (runtime/'run.py').write_text(runner)
    shutil.copyfile(Path(__file__).with_name('logged_features.py'),runtime/'logged_features.py')
    write(MISSING/'selected_case_ids.json',missing)
    post=MISSING/'postprocessing'
    post.mkdir(exist_ok=True)
    for name in ('snapshot.py','verify_snapshot.py'):
        shutil.copyfile(OLD_ASR/'postprocessing'/name,post/name)
    shutil.copyfile(OLD_ASR/'snapshot_validation.json',MISSING/'snapshot_validation.json')
    audit=(OLD_ASR/'postprocessing/audit_export.py').read_text()
    audit=change(audit,"schedule = read(BASE / 'schedule.json')\n    canonical = read(BASE / 'canonical_cases.json')", "selected = set(read(OUTPUT / 'selected_case_ids.json'))\n    assert manifest['selected_case_ids_sha256'] == sha(OUTPUT / 'selected_case_ids.json')\n    schedule = [r for r in read(BASE / 'schedule.json') if r['case_id'] in selected]\n    canonical = [r for r in read(BASE / 'canonical_cases.json') if r['case_id'] in selected]")
    audit=change(audit,"assert len(schedule) == len(canonical) == 1898", "assert len(schedule) == len(canonical) == len(selected)")
    audit=change(audit,"assert entry['case_index'] == case_index", "assert entry['case_id'] == case_id")
    audit=change(audit,"assert len(records) == 1898\n    assert sum(r['payload_exposed'] for r in records) == 1852", "assert len(records) == len(selected)\n    assert all(r['payload_exposed'] for r in records)")
    # This sub-run's audit exports only ASR labels; the merged exporter aligns
    # newly forwarded historical features and canonical features together.
    boundary="    write(OUTPUT / 'case_results.json', records)"
    assert audit.count(boundary)==1
    audit=audit[:audit.index(boundary)]+"    write(OUTPUT / 'case_results.json', records)\n    write(OUTPUT / 'asr_summary.json', summary)\n    write(OUTPUT / 'audit.json', {'status':'passed','cases':len(records),'calls':calls,'official_predicates_rechecked':True})\n\n\nif __name__ == '__main__':\n    main()\n"
    (post/'audit_export.py').write_text(audit)
    write(ROOT/'runtime_binding.json',{'selected_cases':len(missing),
          'runtime_sha256':{p.name:sha(p) for p in runtime.glob('*.py')},
          'postprocessing_sha256':{p.name:sha(p) for p in post.glob('*.py')},
          'plan_sha256':sha(ROOT/'plan.json'), 'selected_case_ids_sha256':sha(MISSING/'selected_case_ids.json')})
    print(serial({'status':'frozen','missing_asr_cases':len(missing),'runtime':str(runtime)}))


if __name__=='__main__':
    main()
