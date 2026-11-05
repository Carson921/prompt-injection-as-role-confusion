"""Audit missing native episodes, merge all results, then verify the archive."""
import argparse
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from reuse_common import *


def run(command,log):
    with (ROOT/log).open('a') as output:
        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    parser.add_argument('--package-only',action='store_true',help='Package an already independently audited full export')
    args=parser.parse_args()
    lock=(ROOT/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(ROOT/'manifest.json')
    expected=manifest['missing_asr_cases']
    total=manifest['total_cases']
    while not args.package_only:
        terminal=read(MISSING/'producer-terminal.json') if (MISSING/'producer-terminal.json').exists() else {}
        if terminal.get('completed')==terminal.get('requested')==expected:
            break
        if not args.watch:
            raise RuntimeError('Missing-case ASR collection is incomplete')
        native=read(MISSING/'progress.json') if (MISSING/'progress.json').exists() else {}
        features=read(ROOT/'feature_progress.json') if (ROOT/'feature_progress.json').exists() else {}
        completed=native.get('completed',0)
        write(ROOT/'progress.json',{'stage':'filling_missing_asr','total':total,
              'reused_asr':total-expected,'new_asr_completed':completed,'asr_completed':total-expected+completed,
              'new_asr_total':expected,'logged_feature_forwards_completed':features.get('completed',0),
              'logged_feature_forwards_total':features.get('total',266),'last_case':native.get('last_case'),
              'continuation_job':read(ROOT/'continuation-job.json').get('job_id') if (ROOT/'continuation-job.json').exists() else None,'updated_at':time.time()})
        time.sleep(30)
    if not args.package_only:
        run([BENCHMARK_PYTHON,'-u',str(MISSING/'postprocessing/audit_export.py')],'missing_audit.log')
        run([NATIVE_PYTHON,'-u',str(Path(__file__).with_name('export_dataset.py'))],'export.log')
    audit=read(ROOT/'dataset_audit.json')
    assert audit['status']=='passed' and audit['full_benchmark_complete'] and audit['completed_asr']==total
    assert audit['case_results_sha256']==sha(ROOT/'case_results.json')
    assert audit['features_receipt_sha256']==sha(ROOT/'probe_inputs/features.json')
    native_audit=read(MISSING/'audit.json')
    assert native_audit['status']=='passed' and native_audit['cases']==expected and native_audit['official_predicates_rechecked']
    excluded={'run.log','finish.log','finish.lock','export.lock','progress.json','launch.json','finisher-launch.json',
              'budget_guard.log','budget_guard_status.json','budget_guard-launch.json','budget_guard.lock',
              'checkpoint-latest.json','launch-management.lock'}
    paths=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in excluded
           and '__pycache__' not in p.parts and p.name!='bundle_files.json' and not p.name.endswith('.tmp')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(ROOT/'bundle_files.json',{'format':'sha256-all-archived-files','files':hashes})
    paths.append(ROOT/'bundle_files.json')
    archive=ROOT.with_suffix('.tar.gz');temp=archive.with_name(archive.name+'.tmp')
    with tarfile.open(temp,'w:gz',compresslevel=1,dereference=True) as tar:
        for p in paths:
            tar.add(p,arcname=ROOT.name+'/'+str(p.relative_to(ROOT)),recursive=False)
    all_hashes={**hashes,'bundle_files.json':sha(ROOT/'bundle_files.json')}
    verified=0
    with tarfile.open(temp,'r:gz') as tar:
        for member in tar:
            assert member.isfile()
            name=str(Path(member.name).relative_to(ROOT.name))
            expected_hash=all_hashes.pop(name)
            stream=tar.extractfile(member);h=hashlib.sha256()
            for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
            assert h.hexdigest()==expected_hash,name
            verified+=1
    assert not all_hashes
    temp.replace(archive)
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    receipt={'status':'complete','archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,
             'cases':total,'reused_asr_cases':total-expected,'new_asr_cases':expected,'all_archive_members_verified':verified,
             'dataset_audit_sha256':sha(ROOT/'dataset_audit.json'),'finished_at':time.time()}
    write(archive.with_name(archive.name+'.complete.json'),receipt)
    write(ROOT/'progress.json',receipt)
    print(serial(receipt),flush=True)


if __name__=='__main__':main()
