"""Metadata-only two-pin diff audit; does not import or run any trial/auditor."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path('/root/prompt-injection-as-role-confusion')
BASE=ROOT/'experiments/role-analysis/outputs/local-gptoss120b'
PREP=BASE/'systemness-complementary-head-readout-preparation/result-auditor'
OLD_SHA='fbd87f808d0913821dfa4391464a658c1ca6069a4566d03b824b1080e27a19c0'
NEW_SHA='0219af8dbf0dc6587f8ba4d4bc1e4f163ae9e745a2b6c4d99a24560a88c8ae85'
PINS={'SOURCE_SHA':'e74e40889ba49ab05815f0faac68c2a522ccc06627b09ff14379929bc53dbdf6','PROTOCOL_SHA':'10c92c0af904b96d42e1ff8c9cbc12f7efb0371da08fdc3962aebd7b8d0d6f54'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=PREP/'source-before-training-pins.py'
current=Path('/tmp/confirmation_review_complementary_readout_results_audit.py')
archived=PREP/'audit.py'
trainer=ROOT/'experiments/role-analysis/systemness_complementary_head_readout.py'
protocol=BASE/'systemness-complementary-head-readout/protocol.json'
assert sha(original)==OLD_SHA
assert sha(current)==sha(archived)==NEW_SHA
before=current.read_bytes();text=before.decode();tree=ast.parse(text);positions=[];found={}
lines=text.splitlines(keepends=True);offsets=[0]
for line in lines:offsets.append(offsets[-1]+len(line))
for node in tree.body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in PINS:
  name=node.targets[0].id;assert name not in found;found[name]=ast.literal_eval(node.value);assert found[name]==PINS[name]
  line=lines[node.value.lineno-1];endline=lines[node.value.end_lineno-1]
  # These two frozen assignment lines are ASCII; AST byte offsets equal chars.
  assert line.isascii() and endline.isascii()
  positions.append((offsets[node.value.lineno-1]+node.value.col_offset,offsets[node.value.end_lineno-1]+node.value.end_col_offset,'None'))
assert found==PINS
restored=text
for start,end,replacement in sorted(positions,reverse=True):restored=restored[:start]+replacement+restored[end:]
assert restored.encode()==original.read_bytes()
assert hashlib.sha256(restored.encode()).hexdigest()==OLD_SHA
# Only frozen SOURCE/protocol bytes are hashed; no model/result/probability IO.
assert sha(trainer)==PINS['SOURCE_SHA'] and sha(protocol)==PINS['PROTOCOL_SHA']
assert current.read_bytes()==before and sha(archived)==NEW_SHA and sha(original)==OLD_SHA
r=dict(stage='metadata_only_complementary640_result_auditor_training_pin_diff_review',passed=True,mustfix=[],
 auditor_source_path=str(current),archived_auditor_path=str(archived),original_auditor_path=str(original),
 auditor_sha256_before_and_after=NEW_SHA,original_auditor_sha256=OLD_SHA,
 changed_constants=PINS,only_two_constant_values_changed=True,reverting_only_the_two_values_to_None_restores_original_byteexact=True,
 frozen_training_source_path=str(trainer),frozen_training_source_sha256=sha(trainer),frozen_protocol_path=str(protocol),frozen_protocol_sha256=sha(protocol),
 no_preflight_rerun=True,actual_auditor_executed=False,training_models_or_results_or_probabilities_read=False,
 row_metadata_or_feature_banks_read=False,test_data_or_results_read=False,gpu_used=False,production_files_changed=False)
p=Path('/tmp/feature_review_complementary_auditor_pin_diff.json');p.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(dict(report=str(p),report_sha256=sha(p),script_sha256=sha(__file__),auditor_sha256=NEW_SHA),indent=2))
