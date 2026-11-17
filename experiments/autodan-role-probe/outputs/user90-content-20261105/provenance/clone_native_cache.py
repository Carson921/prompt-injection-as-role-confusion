"""Clone complete native evidence; rewrite paths, never numerical captures."""
import hashlib
import json
import os
from pathlib import Path
from scorer import sha, write


def clone(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source != destination
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('dataset.json','score_protocol.json'):
        target = destination/name
        if target.exists():
            assert target.read_bytes() == (source/name).read_bytes()
        else:
            target.write_bytes((source/name).read_bytes())
    records=[]
    for path in sorted((source/'scores').glob('*/score.json')):
        value=json.loads(path.read_text())
        if value.get('status')!='complete':
            continue
        assert value.get('complete') is True and len(value['per_case'])==6
        target=destination/'scores'/path.parent.name/'score.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        for index,row in enumerate(value['per_case']):
            old=Path(row['token_artifact']).parent
            new=target.parent/f'case_{index:02d}'
            new.mkdir(exist_ok=True)
            for name,expected in row['artifact_sha256'].items():
                assert sha(old/name)==expected
                if not (new/name).exists():
                    os.link(old/name,new/name)
                assert sha(new/name)==expected
            row['token_artifact']=str(new/'tokens.npz')
            write(new/'score.json',row)
        value['artifact']=str(target)
        write(target,value)
        records.append(dict(source_score_path=str(path),source_score_sha256=sha(path),
                            destination_score_path=str(target),destination_score_sha256=sha(target),
                            template_sha256=sha(value['template']),fitness=value['fitness'],
                            model_forwards_recomputed=0))
    manifest=dict(schema='complete-native-cache-identity-clone-v1',source=str(source),
                  complete_six_task_results=len(records),records=records,
                  preserved_raw_npz_and_request_bytes=True,only_paths_rewritten=True,
                  scorer_protocol_sha256=sha(json.dumps(json.loads((destination/'score_protocol.json').read_text()),sort_keys=True,separators=(',',':'))),
                  exact_input_mask_compatibility_audit_required=True,
                  clone_source_sha256=sha(Path(__file__)))
    write(destination/'native_cache_import.json',manifest)
    print(json.dumps(dict(cloned_complete_candidates=len(records),source=str(source),destination=str(destination))),flush=True)

if __name__=='__main__':
    import sys
    clone(*sys.argv[1:])
