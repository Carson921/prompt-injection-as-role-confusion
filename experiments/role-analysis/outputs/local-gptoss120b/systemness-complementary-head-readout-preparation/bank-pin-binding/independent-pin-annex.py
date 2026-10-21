"""Minimal metadata-only independent review of three final readout bank pins.
No bank, row table, p, model or training/declaration API is loaded.
"""
import ast, copy, hashlib, json, tempfile
from pathlib import Path
ROOT=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
CURRENT=ROOT/'systemness_complementary_head_readout.py'
BEFORE=ROOT/'outputs/local-gptoss120b/systemness-complementary-head-readout-preparation/source-before-bank-pins.py'
NEW_SHA='e74e40889ba49ab05815f0faac68c2a522ccc06627b09ff14379929bc53dbdf6'
OLD_SHA='9c9565f75190f8b81dc97e470dcb54080a9b1229b0b22e174299e8cf138c0a97'
PINNED_COMPLETE='a48ad6d326f1665e6048952d1274b5f880207908ca46bd135d37c787788515e4'
PINNED_AUDIT='e58804dadb6228e5c59a85b19e89c05cfaf4707d8390815067472a3857b89845'
EXTRACTOR_SHA='964335b1dbdbea19219cc4e3db15409195376cd92dfd7125f7d4370758a34509'
EXTRACT_OUT=ROOT/'outputs/local-gptoss120b/systemness-complementary-head-features'
AUDIT=EXTRACT_OUT/'actual-audit/actual-feature-bank-audit.json'
COMPLETE=EXTRACT_OUT/'extraction-complete.json'
NAMES={'EXTRA_COMPLETE_SHA','EXTRA_AUDIT_PATH','EXTRA_AUDIT_SHA'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def assignments(tree):return {n.targets[0].id:n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
checks=[]
def check(name,value):assert value,name;checks.append(name)
check('current-source-exact-hash',digest(CURRENT)==NEW_SHA)
check('before-source-exact-hash',digest(BEFORE)==OLD_SHA)
current,before=CURRENT.read_text(),BEFORE.read_text();ct,bt=ast.parse(current),ast.parse(before);ca,ba=assignments(ct),assignments(bt)
check('same-top-level-assignment-names',set(ca)==set(ba))
changed={k for k in ca if ast.dump(ca[k],include_attributes=False)!=ast.dump(ba[k],include_attributes=False)}
check('only-three-constant-assignments-changed',changed==NAMES)
check('previous-three-pins-were-None',all(ast.literal_eval(ba[k].value) is None for k in NAMES))
values={k:ast.literal_eval(ca[k].value) for k in NAMES}
check('pins-match-root-declared-exact-values',values=={'EXTRA_COMPLETE_SHA':PINNED_COMPLETE,'EXTRA_AUDIT_PATH':str(AUDIT),'EXTRA_AUDIT_SHA':PINNED_AUDIT})
for name in NAMES:ca[name].value=copy.deepcopy(ba[name].value)
check('entire-AST-other-than-three-pins-identical',ast.dump(ct,include_attributes=False)==ast.dump(bt,include_attributes=False))
# Also prove exact text equivalence after reverting only the three assignment lines.
lines=current.splitlines(keepends=True);oldlines=before.splitlines(keepends=True)
for name in NAMES:
    now,old=assignments(ast.parse(current))[name],ba[name]
    assert now.lineno==now.end_lineno==old.lineno==old.end_lineno
    lines[now.lineno-1]=oldlines[old.lineno-1]
check('entire-source-bytes-other-than-three-lines-identical',''.join(lines)==before)
check('completion-file-exact-hash',digest(COMPLETE)==PINNED_COMPLETE)
check('actual-audit-file-exact-hash',digest(AUDIT)==PINNED_AUDIT)
check('extractor-source-exact-hash',digest(ROOT/'systemness_complementary_head_extract.py')==EXTRACTOR_SHA)
done=json.loads(COMPLETE.read_text());audit=json.loads(AUDIT.read_text())
original=[458,462,2045,460,459,2047,456,997];added=[247,2248]
check('actual-independent-audit-terminal-pass-no-tests',audit['stage']=='actual_completed_complementary_head_feature_audit' and audit['audit_passed'] is True and audit['test_data_or_results_read'] is False)
check('audit-extractor-source-and-complete-crossbinding',audit['extractor_source_sha256']==EXTRACTOR_SHA and audit['extractor_complete_sha256']==PINNED_COMPLETE)
check('audit-plan-feature-and-row-hash-crossbinding',all(audit[a]==done[b] for a,b in [('extractor_plan_sha256','plan_sha256'),('features_sha256','features_sha256'),('rows_sha256','rows_sha256')]))
check('complete-native-feature-shape-and-full-coverage',done['rows']==896838 and done['dimensions']==128 and done['feature_shape']==[896838,128] and done['dtype']=='float16' and done['all_rows_written'] is True and done['all_finite'] is True and done['tests_read'] is False)
check('audit-native-feature-shape',audit['rows']==896838 and audit['dimensions']==128 and audit['shape']==[896838,128] and audit['dtype']=='float16')
check('complete-and-audit-ranked-head-identity',audit['original_head_ids']==done['original_head_ids']==original and audit['added_head_ids']==done['added_head_ids']==added)
# bank_gate body must be byte-for-byte frozen from independently reviewed source.
oldgate=next(n for n in bt.body if isinstance(n,ast.FunctionDef) and n.name=='bank_gate')
newgate=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name=='bank_gate')
check('bank-gate-unchanged',ast.dump(oldgate,include_attributes=False)==ast.dump(newgate,include_attributes=False))
first=ast.get_source_segment(current,newgate.body[0]);check('unknown-pins-refused-before-gate-file-hash-or-reads',all(k in first for k in NAMES) and 'is not None' in first and newgate.body[0].value.func.id=='require')
check('source-hash-still-exact-after-review',digest(CURRENT)==NEW_SHA)
folder=Path(tempfile.mkdtemp(prefix='long-data-complementary-readout-pins-audit-',dir='/tmp'))
report=dict(audit_passed=True,scope='Only exact source diff, small completion/audit JSON bytes and source-file hashes. No feature/row bank, p/model/cache body, declare/train/GPU or actual bank_gate execution.',checks=len(checks),check_names=checks,source_sha256=NEW_SHA,before_source_sha256=OLD_SHA,three_pins=values,actual_feature_audit_sha256=PINNED_AUDIT,extractor_complete_sha256=PINNED_COMPLETE,feature_rows=896838,feature_dimensions=128,feature_bytes=896838*128*2,feature_data_read=False,row_map_read=False,model_or_probability_read=False,actual_declare_or_train_run=False,source_diff_only_three_constants=True)
path=folder/'report.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(report=str(path),report_sha256=digest(path),script_sha256=digest(__file__),checks=len(checks),source_sha256=NEW_SHA),indent=2))
