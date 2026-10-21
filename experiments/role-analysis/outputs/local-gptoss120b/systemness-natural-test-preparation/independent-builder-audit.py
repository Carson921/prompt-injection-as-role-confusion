"""Independent synthetic CPU natural-test source audit; no test replies/results."""
from collections import defaultdict
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT=Path('/root/prompt-injection-as-role-confusion')
sys.path.insert(0,str(ROOT/'experiments/role-analysis'))
b=importlib.import_module('systemness_natural_test')
r=importlib.import_module('systemness_natural_test_confirm')
folder=Path(tempfile.mkdtemp(prefix='confirmation-review-natural-test-',dir='/tmp'))
checks=[]
source_paths=[Path(b.__file__),Path(r.__file__)]
start_sha={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
formal_paths=[Path(b.OUT),Path(b.CACHE),Path(r.OUT),Path(r.CACHE)]
formal_before={str(path):path.exists() for path in formal_paths}

def check(name,condition=True):
    if not condition:raise AssertionError(name)
    checks.append(name)

def reject(name,fn):
    try:fn()
    except (RuntimeError,AssertionError,ValueError,KeyError,FileNotFoundError):checks.append(name)
    else:raise AssertionError('Expected rejection: '+name)

class Union:
    def __init__(self,nodes):self.parent={node:node for node in nodes}
    def find(self,node):
        while self.parent[node]!=node:
            self.parent[node]=self.parent[self.parent[node]];node=self.parent[node]
        return node
    def join(self,nodes):
        nodes=list(nodes)
        for node in nodes[1:]:self.parent[self.find(node)]=self.find(nodes[0])
    def groups(self):
        groups=defaultdict(set)
        for node in self.parent:groups[self.find(node)].add(node)
        return {frozenset(nodes) for nodes in groups.values()}

# Fixed128/32 synthetic design with direct, internal and transitive reference links.
ledger=[dict(source_uid=f'test:{i:03d}',family_id=f'family:{i//4:02d}',status='prepared' if i<126 else 'quality-failed') for i in range(128)]
assistant=[dict(message_uid=row['source_uid']+'/message2',source_uid=row['source_uid'],split='test',message_ix=2,
    keys={('full',f'{i+100:064x}')}) for i,row in enumerate(ledger) if row['status']=='prepared']
a,bkey,c,d,repeat,oldkey=[('normalized',letter*64) for letter in 'abcdef']
assistant[0]['keys'].add(a);assistant[4]['keys'].update([bkey,c]);assistant[8]['keys'].update([d,repeat]);assistant[12]['keys'].add(repeat)
for i in range(28,32):assistant[i]['keys'].add(oldkey)
reference=[dict(message_uid='dev:r1/m',source_uid='dev:r1',split='fit',message_ix=2,keys={a,bkey}),
           dict(message_uid='dev:r2/m',source_uid='dev:r2',split='validation',message_ix=2,keys={c,d})]
old_index={oldkey:{'archived-assistant'}}
expected_old=set(old_index)
expected_ref=set().union(*(row['keys'] for row in reference))
all_messages=assistant+reference
all_nodes={row['message_uid'] for row in all_messages}|{'old-reference:archived-assistant'}
key_nodes=defaultdict(set)
for row in all_messages:
    for key in row['keys']:key_nodes[key].add(row['message_uid'])
for key,refs in old_index.items():key_nodes[key].update('old-reference:'+ref for ref in refs)
message_union=Union(all_nodes)
for nodes in key_nodes.values():message_union.join(nodes)
roots=Union([row['source_uid'] for row in ledger if row['status']=='prepared'])
families=defaultdict(list)
for row in ledger:
    if row['status']=='prepared':families[row['family_id']].append(row['source_uid'])
for nodes in families.values():roots.join(nodes)
message_roots={row['message_uid']:row['source_uid'] for row in assistant}
for group in message_union.groups():roots.join(message_roots[node] for node in group if node in message_roots)
expected_groups=roots.groups()
expected_clean={row['source_uid'] for row in assistant if not row['keys']&expected_old and not row['keys']&expected_ref}
expected_clean_families={row['family_id'] for row in ledger if row['source_uid'] in expected_clean}
audit=b.duplicate_audit(ledger,assistant,reference,old_index,dict(synthetic=True))
actual_groups={frozenset(group['source_uids']) for group in audit['components']}
check('independent union agrees on128-ledger family/internal/transitive-reference components',actual_groups==expected_groups)
check('direct old/dev exclusions match independent key intersection',{row['source_uid'] for row in ledger if row['eligible_clean_test']}==expected_clean)
check('clean subset retains original component union',{frozenset(group['clean_source_uids']) for group in audit['components'] if group['clean_source_uids']}=={group&expected_clean for group in expected_groups if group&expected_clean})
check('all128 fixed IDs and two failures preserved',len(ledger)==128 and audit['eligibility_counts']['quality_failed']==2)
check('indirect reference collision remains clean per declared direct-only rule',ledger[12]['eligible_clean_test'])
check('whole dirty family absent from clean effective family count',len(expected_clean_families)==31)

# All-quality-failed construction must retain ledger without invented observations.
records=[dict(source_uid=f'failed:{i}',dataset='fixture',native_id=str(i),split='test',revision='synthetic',
    root_raw_sha256='fixture',template_id=f't{i//4}',family_id=f'f{i//4}',source_snapshot_sha256='fixture',
    messages=[],failure='synthetic-empty-response') for i in range(128)]
inputs,rows,tokens,failed,duplicates,coverage=b.build_dataset(records,None,[],{},dict(synthetic=True))
check('all-failed test still retains128 statuses and zero observations',len(failed)==128 and not len(inputs) and not len(rows) and tokens==[] and duplicates['eligibility_counts']['quality_failed']==128)
b.verify_cached_content(inputs,rows,tokens,failed,None)
check('all-failed empty schemas pass cache identity verifier')
check('all-failed reports zero supported roles',all(value==0 for value in coverage['role_counts'].values()) and coverage['input_prompts']==0 and coverage['content_rows']==0)

# Pure pinned model validator: no filesystem/model/source calls.
frozen=dict(models={name:'0'*64 for name in r.NAMES},thresholds={name:.5 for name in r.NAMES},
    primary_candidate=r.PRIMARY_CANDIDATE,primary_validation_promoted=True,role_order=b.data.ROLES,
    selected_head_ids=[458,462,2045,460,459,2047,456,997])
frozen['models'][r.PRIMARY]=r.PRIMARY_MODEL_SHA256;frozen['thresholds'][r.PRIMARY]=r.PRIMARY_THRESHOLD
r.validate_frozen({name:folder/(name+'.npz') for name in r.NAMES},frozen['thresholds'],frozen)
check('pure primary validator accepts exact pinned head128')
for field,value in [('model','1'*64),('threshold',.5),('candidate','joint-mlp64')]:
    wrong=deepcopy(frozen)
    if field=='model':wrong['models'][r.PRIMARY]=value
    elif field=='threshold':wrong['thresholds'][r.PRIMARY]=value
    else:wrong['primary_candidate']=value
    reject('pinned primary rejects '+field,lambda wrong=wrong:r.validate_frozen({name:folder/(name+'.npz') for name in r.NAMES},wrong['thresholds'],wrong))

# A failed model gate cannot create OUT or enter a reply reader.
from unittest.mock import patch
never=folder/'never-created'
seen=[]
def failed_snapshot():
    seen.append('model-gate');raise RuntimeError('synthetic missing promoted exact model')
def forbidden_reader(*args,**kwargs):raise AssertionError('Reply reader must not run')
with patch.object(b,'OUT',never),patch.object(b,'CACHE',folder/'never-cache'),patch.object(b,'source_context',lambda:{}),patch.object(b,'frozen_snapshot',failed_snapshot),patch.object(b,'read_sources',forbidden_reader):
    reject('model gate fails before directory or assistant read',b.prepare)
check('failed model gate creates no formal or temporary preparation namespace',seen==['model-gate'] and not never.exists())

# Final hash/formal-state checks; the caller reruns if an owner edits during audit.
check('formal natural-test directories unchanged',formal_before=={str(path):path.exists() for path in formal_paths})
end_sha={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
check('both new sources unchanged during independent CPU audit',start_sha==end_sha)
report=dict(source_sha256=start_sha,checks=checks,checks_passed=len(checks),
    synthetic_ledger_roots=128,all_dependency_components=len(expected_groups),clean_dependency_components=sum(bool(group&expected_clean) for group in expected_groups),
    all_system_families=len(families),clean_system_families=len(expected_clean_families),
    clean_roots=len(expected_clean),fixture_directory=str(folder),formal_paths=formal_before,
    gpu_used=False,real_test_replies_or_results_read=False,c4_confirmation_data_read=False,formal_outputs_written=False)
out=Path('/tmp/confirmation_review_natural_test_audit.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(out);print(json.dumps(report,sort_keys=True))
