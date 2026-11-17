import copy, importlib.util, json, shutil, tempfile, unittest
from datetime import datetime, timezone
from pathlib import Path

script = Path('experiments/role-analysis/outputs/deepseek-v4-flash-agentdojo389-plain-user-delimiter-20261115/asr-supplement/prepare_asr_interpretation.py').resolve()
spec = importlib.util.spec_from_file_location('asr_supplement', script)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
checks = []

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')

def read(path): return json.loads(path.read_text())

def rebind(root, *, case_id=None, native_row=False):
    native = read(root/'native-evaluation-audit.json')
    if case_id:
        case = read(root/'cases'/(case_id+'.json'))
        native['cases_sha256'][case_id] = m.file_sha(root/'cases'/(case_id+'.json'))
        if native_row:
            row = next(r for r in native['native_results'] if r['case_id']==case_id)
            row.update(security=case['security'], utility=case['utility'])
    write(root/'native-evaluation-audit.json', native)
    validation = read(root/'independent-validation.json')
    validation['artifact_sha256'] = {name:m.file_sha(root/name) for name in ('schedule.json','summary.json','native-evaluation-audit.json')}
    write(root/'independent-validation.json', validation)

def fixture(root, configurations=((383,100),(380,80))):
    schedule, cases, rows, hashes = [], [], [], {}
    for index, case_id in enumerate(m.canonical_ids()):
        condition, suite, user, attack = case_id.split('/')
        n,s = configurations[index//389]
        local = index % 389
        item = dict(case_id=case_id,condition=condition,base_case_id='/'.join(case_id.split('/')[1:]),
                    suite=suite,user_task_id=user,injection_task_id=attack,kind='attack',actual_goal='synthetic fixture')
        schedule.append(item)
        record = dict(item, status='complete' if local<n else 'error',
                      security=(local<s) if local<n else None, utility=False if local<n else None, requests=[])
        if local>=n:
            error = {'category':'model','type':'IncompleteNativeCompletion','message':'synthetic native budget error'}
            record['error']='NativeModelError: '+json.dumps(error)
            record['requests']=[{'status':'native_error','error':json.dumps(error),'call_id':case_id+'/call_0000'}]
        path=root/'cases'/(case_id+'.json');write(path,record)
        hashes[case_id]=m.file_sha(path)
        rows.append({'case_id':case_id,'valid':True,'security':record['security'],'utility':record['utility'],
                     'native_error_reproduced':local>=n})
        cases.append(record)
    metrics={}
    for condition,(n,s) in zip(m.CONDITIONS,configurations):
        metrics[condition]={'expected_cases':389,'terminal_cases':389,'scored_cases':n,'error_cases':389-n,
                           'missing_cases':0,'attack_successes':s,'utility_successes':0,'data_fully_scored':n==389,
                           'asr':s/389 if n==389 else None,'utility':0.0 if n==389 else None,
                           'asr_scored':s/n if n else None,'utility_scored':0.0 if n else None}
    fully=all(n==389 for n,s in configurations)
    write(root/'schedule.json',schedule)
    write(root/'summary.json',{'execution_complete':True,'scheduled_cases':778,'data_fully_scored':fully,'conditions':metrics})
    write(root/'native-evaluation-audit.json',{'status':'passed','complete':True,'scheduled_cases':778,'audited_cases':778,
          'schedule_sha256':m.file_sha(root/'schedule.json'),'model_generation_performed':False,
          'cases_sha256':hashes,'native_results':rows})
    write(root/'independent-validation.json',{'status':'passed','complete':True,'validator_sha256':m.FROZEN_VALIDATOR_SHA256,
          'checks':[{'name':'synthetic_already_validated_fixture','ok':True}],'missing_artifacts':[],
          'metrics':metrics,'data_fully_scored':fully,'artifact_sha256':{name:m.file_sha(root/name) for name in
            ('schedule.json','summary.json','native-evaluation-audit.json')}})
    return schedule

class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=tempfile.TemporaryDirectory(prefix='asr-supplement-guards-',dir='/dev/shm')
        cls.template=Path(cls.base.name)/'template';fixture(cls.template)
    @classmethod
    def tearDownClass(cls):cls.base.cleanup()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=self.base.name,prefix='case-')
        self.root=Path(self.temp.name)/'run';shutil.copytree(self.template,self.root)
    def tearDown(self):self.temp.cleanup()
    def refuse(self, phrase):
        with self.assertRaisesRegex(ValueError,phrase):m.prepare(self.root)
        self.assertFalse((self.root/m.OUTPUT_NAME).exists())
    def test_valid_final_exact_bounds_nulls_and_idempotence(self):
        result=m.prepare(self.root);p=result['conditions']['plain'];d=result['conditions']['plain_delimiter']
        self.assertIsNone(p['ASR_all389']);self.assertEqual(p['ASR_scored'],100/383)
        self.assertEqual((p['n_scored'],p['s_successes'],p['e_unscored_errors'],p['m_missing_episodes']),(383,100,6,0))
        self.assertEqual(p['identification_interval']['lower']['exact'],'100/389')
        self.assertEqual(p['identification_interval']['upper']['exact'],'106/389')
        self.assertEqual(d['identification_interval']['upper']['value'],89/389)
        self.assertEqual(result['delimiter_minus_plain_identification_interval']['lower']['exact'],'-26/389')
        self.assertEqual(result['delimiter_minus_plain_identification_interval']['upper']['exact'],'-11/389')
        self.assertEqual(p['unscored_error_case_counts_by_type'],{'IncompleteNativeCompletion':6})
        self.assertTrue(all(row['security'] is row['utility'] is None for row in p['unscored_errors']))
        self.assertEqual(len(result['bindings']['cases_sha256']),778)
        self.assertEqual(len(result['bindings']['total_files_sha256']),4)
        output=self.root/m.OUTPUT_NAME;before=output.read_bytes();stamp=output.stat().st_mtime_ns
        self.assertEqual(m.prepare(self.root),result);self.assertEqual(output.read_bytes(),before)
        self.assertEqual(output.stat().st_mtime_ns,stamp)
    def test_partial_execution_refused(self):
        summary=read(self.root/'summary.json');summary['execution_complete']=False;write(self.root/'summary.json',summary)
        rebind(self.root);self.refuse('778 formal episodes must finish')
    def test_both_final_audits_required(self):
        for name in ('independent-validation.json','native-evaluation-audit.json'):
            original=read(self.root/name);changed=copy.deepcopy(original);changed['status']='running';changed['complete']=False
            write(self.root/name,changed);rebind(self.root);self.refuse('complete passed audit')
            write(self.root/name,original);rebind(self.root)
    def test_null_error_cannot_be_relabelled_false(self):
        case_id=m.canonical_ids()[383];path=self.root/'cases'/(case_id+'.json')
        record=read(path);record['security']=False;write(path,record);rebind(self.root,case_id=case_id,native_row=True)
        self.refuse('retain null security and utility')
    def test_case_raw_sha_change_refused(self):
        path=self.root/'cases'/(m.canonical_ids()[0]+'.json');path.write_bytes(path.read_bytes()+b' ')
        self.refuse('Case changed since native audit')
    def test_total_raw_sha_change_refused(self):
        path=self.root/'summary.json';path.write_bytes(path.read_bytes()+b' ')
        self.refuse('Final validation SHA does not match')
    def test_native_outcome_change_is_not_hidden_by_rebinding(self):
        native=read(self.root/'native-evaluation-audit.json');native['native_results'][0]['security']=False
        write(self.root/'native-evaluation-audit.json',native);rebind(self.root)
        self.refuse('Native audited outcomes differ')
    def test_duplicate_or_missing_389x2_case_is_refused(self):
        schedule=read(self.root/'schedule.json');schedule[-1]=schedule[0];write(self.root/'schedule.json',schedule);rebind(self.root)
        self.refuse('canonical unique 389 x 2')
    def test_duplicate_native_result_is_refused(self):
        native=read(self.root/'native-evaluation-audit.json');native['native_results'][-1]=native['native_results'][0]
        write(self.root/'native-evaluation-audit.json',native);rebind(self.root)
        self.refuse('778 unique case results')
    def test_extra_case_and_stale_metrics_are_refused(self):
        extra=self.root/'cases/extra.json';write(extra,{})
        self.refuse('missing, extra or duplicate case files');extra.unlink()
        case_id=m.canonical_ids()[100];path=self.root/'cases'/(case_id+'.json')
        record=read(path);record['security']=True;write(path,record);rebind(self.root,case_id=case_id,native_row=True)
        self.refuse('case counts/outcomes differ from final metrics')
    def test_exact_bounds_at_all_scored_and_all_unscored_edges(self):
        fixture(self.root,configurations=((389,0),(389,389)))
        result=m.prepare(self.root)
        for condition,value in [('plain',0),('plain_delimiter',1)]:
            stats=result['conditions'][condition]
            self.assertEqual(stats['ASR_all389'],value)
            self.assertIn('Equals the full 389-case ASR',stats['ASR_scored_scope'])
            self.assertTrue(result['interpretation']['point_identified_by_condition'][condition])
            self.assertEqual(stats['identification_interval']['lower']['value'],value)
            self.assertEqual(stats['identification_interval']['upper']['value'],value)
        (self.root/m.OUTPUT_NAME).unlink()
        fixture(self.root,configurations=((0,0),(0,0)))
        result=m.prepare(self.root)
        for stats in result['conditions'].values():
            self.assertIsNone(stats['ASR_all389']);self.assertIsNone(stats['ASR_scored'])
            self.assertIn('Conditional on native-scored cases only',stats['ASR_scored_scope'])
            self.assertEqual(stats['identification_interval']['lower']['value'],0)
            self.assertEqual(stats['identification_interval']['upper']['value'],1)
        self.assertFalse(any(result['interpretation']['point_identified_by_condition'].values()))
        self.assertIn('interval collapses to the exact point', result['interpretation']['lower_bound_interpretation'])
        self.assertEqual(result['delimiter_minus_plain_identification_interval']['lower']['value'],-1)
        self.assertEqual(result['delimiter_minus_plain_identification_interval']['upper']['value'],1)
    def test_existing_different_output_and_midread_changes_refused(self):
        output=self.root/m.OUTPUT_NAME;output.write_text('{"existing":"different"}\n');before=output.read_bytes()
        with self.assertRaisesRegex(ValueError,'Existing supplement differs'):m.prepare(self.root)
        self.assertEqual(output.read_bytes(),before);output.unlink()
        result=m.build_supplement(self.root)
        path=self.root/'cases'/(m.canonical_ids()[0]+'.json');path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'Case changed before publication'):m.assert_snapshot_unchanged(self.root,result)
        self.assertFalse(output.exists())

class RecordedResult(unittest.TextTestResult):
    def addSuccess(self,test):
        checks.append({'name':test.id().split('.')[-1],'ok':True});super().addSuccess(test)

suite=unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests)
result=unittest.TextTestRunner(verbosity=2,resultclass=RecordedResult).run(suite)
if not result.wasSuccessful():raise SystemExit(1)
formal=script.parents[1]
require_missing=['summary.json','independent-validation.json','native-evaluation-audit.json']
assert not (formal/m.OUTPUT_NAME).exists()
record={'format_version':'asr-supplement-guard-validation-v1','status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'generator_sha256':m.file_sha(script),'frozen_report_sha256':m.file_sha(formal.parents[1]/'deepseek_agentdojo_report.py'),
        'scope':'Synthetic temporary fixtures plus actual partial-run refusal only; no final ASR produced',
        'checks':checks,'tests_run':result.testsRun,'partial_formal_run':{'expected_exit_code':1,'observed_exit_code':1,
          'reason':'Final prerequisite is absent: summary.json','final_output_exists':False},
        'new_generation_or_evaluation':False,'http_or_gpu_used':False,'synthetic_fixture_directories_removed':True}
# The frozen source path is at experiments/role-analysis, two levels above RUN.
record['frozen_report_sha256']=m.file_sha(formal.parent.parent/'deepseek_agentdojo_report.py')
write(script.parent/'guard-validation.json',record)
print(json.dumps({'tests_run':result.testsRun,'guard_record':str(script.parent/'guard-validation.json'),'script_sha256':m.file_sha(script)}))