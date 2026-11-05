"""Read-only validation, live snapshots and safe preparation for later resumption."""
import argparse
from reuse_common import *
from checkpoint_state import process_info,native_status,verify_frozen_binding,create_checkpoint,prepare_resume


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','snapshot','prepare-resume']);a=p.parse_args()
    if a.action=='status':
        binding=verify_frozen_binding();s=native_status()
        print(serial({'status':'valid','native_binding':binding,'completed_asr':558+len(s['completed']),
              'pending_asr':len(s['pending']),'worker':process_info(read(ROOT/'launch.json'))}));return
    if a.action=='snapshot':path,value=create_checkpoint('Live checkpoint verified during current allocation')
    else:path,value=prepare_resume()
    print(serial({'checkpoint':str(path),'status':value['status'],'completed_asr':value['completed_asr'],'pending_asr':value['pending_asr']}))


if __name__=='__main__':main()
