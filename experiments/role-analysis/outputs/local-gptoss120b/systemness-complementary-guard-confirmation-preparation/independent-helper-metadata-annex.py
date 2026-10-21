"""One root-authorized published-JSON schema fixture; no actual snapshot."""
import ast,copy,hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

HERE=Path('/root/prompt-injection-as-role-confusion/experiments/role-analysis')
SOURCE=HERE/'systemness_complementary_guard_confirm_snapshot.py'
SOURCE_SHA='9b3f2afdda695e1e6f5ff74952877af36f559a6320069fe3d5b9ff2781cb3e21'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    assert sha(SOURCE)==SOURCE_SHA
    tree=ast.parse(SOURCE.read_text());m={'__file__':str(SOURCE),'__name__':'isolated_metadata_fixture'};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef))],type_ignores=[]),str(SOURCE),'exec'),m)
    cal=m['CALIBRATION'];train=m['TRAINER'];auditpath=cal/'actual-audit/actual-calibration-audit.json'
    names=[cal/n for n in ['protocol.json','complete.json','selection.json','threshold-ledger.json']]+[train/n for n in ['protocol.json','records.json','zero-controls.json','controls.json','selection.json','actual-audit/actual-development-audit.json']]+[auditpath]
    values={str(p):json.loads(p.read_text()) for p in names};audit=values[str(auditpath)]
    pins=dict(CALIBRATION_SOURCE_SHA='70c6ae046cbbb5485047733b593afe23a1d61376f047b2ad29e75c692e7bb85d',CALIBRATION_PROTOCOL_SHA='7edc1b5ba2c55a21ebe01bc5af9f850c4865a933bb9bc4b1274520d6b00ec64a',CALIBRATION_COMPLETE_SHA='9d5b032c1b055eda8ccb5620620f873b2e4775cb533729b787d86436857c2e94',CALIBRATION_DECISION_SHA='e220bda41077f5931dc81b951f10e55e1160904e303f3299a89dfdc300430a0c',CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA='3296c36da7efffe9ab45a75c12d8ebf1640b18f126cd64166674ae19002bc323',CALIBRATION_ACTUAL_AUDIT_PATH=auditpath,CALIBRATION_ACTUAL_AUDIT_SHA='1a6e545db44a2d76c8da30886409d7c3b8376e670281a130b464a2e8821f4667')
    assert sha(auditpath)==pins['CALIBRATION_ACTUAL_AUDIT_SHA']
    m.update(pins);hashes=dict(audit['provenance_sha256']);hashes[str(auditpath)]=pins['CALIBRATION_ACTUAL_AUDIT_SHA'];hashes[audit['auditor_source_path']]=pins['CALIBRATION_APPROVED_AUDITOR_SOURCE_SHA'];access=[]
    def fake_read(path):
        key=str(path);assert key in values,'Forbidden nonwhitelisted JSON: '+key;access.append(key);return copy.deepcopy(values[key])
    def fake_digest(path):
        key=str(path);assert key in hashes,'Forbidden unclosed byte access: '+key;return hashes[key]
    def forbidden(*args,**kwargs):raise AssertionError('Model/scaler/probability NPZ load forbidden')
    m['read']=fake_read;m['digest']=fake_digest
    index={x['record']:x for x in values[str(train/'records.json')]}
    with patch.object(np,'load',forbidden):result=m['calibration_material']({},values[str(train/'actual-audit/actual-development-audit.json')],values[str(train/'protocol.json')],index=index)
    assert result['snapshot_certificate']['eligible'] is True and result['snapshot_certificate']['original_trial_promoted'] is False
    assert result['snapshot_certificate']['system_threshold']==values[str(cal/'selection.json')]['system_threshold']
    assert sha(SOURCE)==SOURCE_SHA
    output=Path(tempfile.mkdtemp(prefix='confirmation-review-guard-snapshot-metadata-',dir='/tmp'))/'report.json'
    report=dict(stage='published_JSON_only_guard_snapshot_adapter_schema_audit',audit_passed=True,helper_source_sha256=SOURCE_SHA,calibration_actual_audit_sha256=pins['CALIBRATION_ACTUAL_AUDIT_SHA'],original_actual_audit_sha256=m['ACTUAL_AUDIT_SHA'],checks_passed=1,published_JSON_whitelist_read=True,JSON_paths_read=sorted(set(access)),digest_callback_uses_certified_map_only=True,np_load_forbidden=True,actual_model_scaler_probability_rowtable_rawbank_tokenizer_ID_body_read=False,snapshot_declared=False,gpu_used=False,certificate=result['snapshot_certificate'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(report_path=str(output),report_sha256=sha(output),script_sha256=sha(__file__))),flush=True)
if __name__=='__main__':main()
