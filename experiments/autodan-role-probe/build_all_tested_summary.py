"""Join saved ASR observations and six-role probabilities without model calls."""
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent / 'outputs/all-tested-template-summary-20261105'
TZ = ZoneInfo('America/New_York')
ROLES = ['system', 'developer', 'user', 'cot', 'assistant', 'tool']
ORDER = [
    ('A', '95556efd7d3d6d4d', '短模板 A', '本轮短模板'),
    ('B', 'bff233abbe7c9945', '短模板 B', '本轮短模板'),
    ('C', 'c9b524c03f34523a', '短模板 C', '本轮短模板'),
    ('D', '683ffe58c655c1a4', '短模板 D', '本轮短模板'),
    ('E', '918d98aff2d495b3', '短模板 E', '本轮短模板'),
    ('F', '803856d01873aa07', '短模板 F', '本轮短模板'),
    ('H1', '2ac375c30249a276', 'Pilot', '历史优化模板'),
    ('H2', '304e67b75577cf4f', 'Near-delimiter', '历史优化模板'),
    ('H3', '9f3a23376f138fbb', 'CoT-refined', '历史优化模板'),
    ('H4', 'e2c5c629061badce', '历史长重复模板', '历史优化模板'),
    ('REF-P', 'f9d007dadefae749', 'plain', '基线参考'),
    ('REF-R', '128b9c61bf6c95fd', 'remember', '基线参考'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def pointer(value, path):
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def local_time(value):
    return datetime.fromisoformat(value).astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')


def run():
    ap, pp = ROOT/'asr_inventory.json', ROOT/'role_probability_inventory.json'
    assert sha(ap) == '9839ffd71095f1dc9ee97fe4a2f092503bc7b899a5b08ec53d74be1fbc8788ca'
    assert sha(pp) == '93c0874136b225516dce43f581419a8b2bcd82cbfb0888309806b6aaff3fde38'
    asr, probs = load(ap), load(pp)
    assert probs['role_order'] == ROLES
    assert probs['current_prepare_verification']['current_case_preparations'] == 72
    template_map = {r['raw_utf8_sha256']: r for r in
                    asr['canonical_optimized_templates'] + asr['baseline_templates']}
    prob_map = {r['template_raw_sha256']: r for r in probs['fixed_six_rows']}
    observations = {r['observation_id']: r for r in asr['observations']}
    assert set(template_map) == set(prob_map) and len(template_map) == 12
    run_order = sorted(asr['runs'], key=lambda r:r['started_at_unix'])
    run_ids = {r['run_id']: 'R'+str(i+1) for i,r in enumerate(run_order)}
    summary, cases, template_lines, exports, mapping = [], [], [], [], {}
    output = ROOT/'templates'
    output.mkdir(exist_ok=False)
    for identifier, prefix, name, category in ORDER:
        matches = [raw for raw in template_map if raw.startswith(prefix)]
        assert len(matches) == 1
        raw = matches[0]
        row, probability = template_map[raw], prob_map[raw]
        source = row['template_source']
        assert sha(source['path']) == source['sha256']
        text = pointer(load(source['path']), source['json_pointer'])
        assert isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == raw
        assert text.count('{goal}') == 1
        score = probability['source_score']
        assert sha(score['path']) == score['sha256']
        assert load(score['path'])['template'] == text
        assert probability['current_complete_goal_mask_verified']
        assert probability['all_saved_masks_equal_current_complete_masks']
        assert probability['complete_per_case_count'] == 6
        dist = probability['role_probabilities']
        assert set(dist) == set(ROLES) and abs(math.fsum(dist.values())-1) < 1e-6
        for role in ROLES:
            assert abs(math.fsum(c['goal_role_probabilities'][role] for c in probability['per_case'])/6-dist[role]) < 1e-12
        m = row['canonical_metrics']
        obs = observations[row['canonical_observation_id']]
        assert m == obs['metrics'] and m['scheduled'] == m['terminal'] == 389
        assert obs['audit_covers_all_current_terminal_cases']
        assert m['scored']+m['unknown'] == 389
        assert abs(m['asr_lower_bound']-m['successes']/389) < 1e-12
        assert abs(m['asr_upper_bound']-(m['successes']+m['unknown'])/389) < 1e-12
        path = output/f'{identifier}_{raw[:16]}.txt'
        path.write_bytes(text.encode())
        assert sha(path) == raw
        mapping[raw] = identifier
        note = '历史长重复模板，不满足当前短模板去重复约束。' if identifier == 'H4' else ''
        if identifier == 'H4':
            note += '旧选择元数据顶层为 error；引用的 native score 为 complete，六任务均已核验。'
        summary.append(dict(template_id=identifier, name=name, category=category,
            template_kind=row['template_kind'], original_optimization_target=probability['original_optimization_target_role'],
            wrapper_tokens=probability['wrapper_tokens'], role_probabilities=dist,
            probe_case_count=6, role_scope='固定六任务，完整 goal token 均值，再对任务等权平均',
            canonical_run=run_ids[obs['run_id']], canonical_observation_id=row['canonical_observation_id'],
            asr_started_local=local_time(obs['started_at_utc']), asr=m,
            all_asr_observation_ids=[row['canonical_observation_id'], *row['alternate_observation_ids']],
            template_raw_sha256=raw, template_file=str(path), template_source=source,
            probe_score_source=score, asr_audit_source=obs['asr_audit'], notes=note))
        restored = []
        for number, line in enumerate(text.splitlines(keepends=True), 1):
            if line.endswith('\r\n'):
                body, ending = line[:-2], '\r\n'
            elif line[-1:] in ('\n','\r','\v','\f','\x1c','\x1d','\x1e','\x85','\u2028','\u2029'):
                body, ending = line[:-1], line[-1:]
            else:
                body, ending = line, ''
            labels = {'\n':'LF', '\r\n':'CRLF', '\r':'CR', '':'EOF'}
            template_lines.append(dict(template_id=identifier, line_number=number,
                text=body, line_ending=ending, line_ending_label=labels.get(ending, repr(ending))))
            restored.append(body+ending)
        assert ''.join(restored) == text
        exports.append(dict(template_id=identifier, raw_sha256=raw, path=str(path),
                            characters=len(text), utf8_bytes=len(text.encode())))
        for case in probability['per_case']:
            parts = case['case_id'].split('/')
            cases.append(dict(template_id=identifier, suite=parts[-3], user_task=parts[-2],
                injection_task=parts[-1], source_case_id=case['case_id'], goal_tokens=case['goal_tokens'],
                role_probabilities=case['goal_role_probabilities'], input_sha256=case['input_sha256'],
                token_artifact=case['token_artifact'], current_complete_goal_mask_verified=True))
    runs = []
    canonical_ids = {r['canonical_observation_id'] for r in summary}
    for obs in sorted(asr['observations'], key=lambda r:(r['started_at_utc'], mapping[r['raw_utf8_sha256']])):
        m = obs['metrics']
        assert m['scored']+m['unknown'] == m['scheduled']
        assert m['terminal'] == m['scored']+m['native_errors']
        assert m['terminal']+m['missing'] == m['scheduled']
        imported = 54 if 'eval_best6389_optimized_only' in obs['run_id'] else 0
        shared_source = 54 if 'eval_best6389__' in obs['observation_id'] and obs['condition']=='optimized' else 0
        runs.append(dict(run_id=run_ids[obs['run_id']], template_id=mapping[obs['raw_utf8_sha256']],
            condition=obs['condition'], observation_id=obs['observation_id'],
            template_kind=obs['template_kind'], run_label=obs['run_id'],
            started_local=local_time(obs['started_at_utc']), canonical=obs['observation_id'] in canonical_ids,
            population=m['population'], metrics=m, audit_status=obs['asr_audit_status'],
            all_terminal_records_audited=obs['audit_covers_all_current_terminal_cases'],
            imported_attempts=imported, source_attempts_also_in_completed_run=shared_source,
            template_raw_sha256=obs['raw_utf8_sha256'], template_source=obs['template_source'],
            audit_source=obs['asr_audit'], summary_source=obs['summary']))
    assert len(summary)==12 and len(cases)==72 and len(runs)==21
    data = dict(schema='all-ASR-tested-templates-reporting-data-v1',
        title='已测试 ASR 模板的 goal 角色概率总表', as_of_date='2026-11-05',
        timezone='America/New_York', role_order=ROLES, probabilities_unit='fraction',
        optimized_template_count=10, baseline_template_count=2,
        representative_run_selection_rule='主表每个原文 SHA 采用该模板最早一轮经审计的完整389任务结果；不按最高ASR挑选。其他完整批次及中止批次均保留在Runs。',
        missing_result_label='n.a.', summary_rows=summary, asr_runs=runs,
        probe_cases=cases, template_lines=template_lines, template_exports=exports,
        notes=[
            'P(role) 均为同一 checkpoint、同一固定六任务模拟上下文下，第一份原始 payload 中完整 goal token 的均值，再对六任务等权平均。',
            '六个角色顺序为 system、developer、user、cot、assistant、tool；数值存储为0到1，工作簿显示为百分比。未对浮点和接近1的误差强制归一化。',
            'Wrapper token 数不包含 goal。历史长重复模板保留用于完整报告，不视为满足当前短模板约束。',
            'ASR下界=成功/计划总数；ASR上界=(成功+未知)/计划总数；已评分ASR=成功/已评分。完整批次的错误保持未知，中止批次另含未执行任务。',
            '不同批次不能直接合并为一个ASR；54条中止批次的optimized记录后来原样复用到完整批次，Runs保留两处观测但不重复合并。',
            'Templates按原文逐行保存，行号和换行类型可还原全部模板；外部txt副本与原始模板UTF8 SHA完全一致。',
            '原目录20261027的plain/prerequisite只有历史汇总且缺少原始payload绑定，未按名称合并到这里的固定六任务数据。',
        ],
        dataset_sha256=probs['dataset_sha256'], checkpoint_sha256=probs['probe_checkpoint_sha256'],
        counts=asr['counts'], source_bindings={str(ap):sha(ap),str(pp):sha(pp)},
        current_prepare_verification=probs['current_prepare_verification'],
        excluded_not_asr_tested=asr['excluded_not_asr_tested'],
        legacy_references_separate_scope=probs['legacy_historical_baseline_references_separate_scope'],
        new_model_predictions=0, new_API_calls=0, new_ASR_runs=0)
    path = ROOT/'reporting_data.json'
    with path.open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path), summary_rows=12, optimized=10,
        run_rows=21, probe_rows=72, template_lines=len(template_lines),
        maximum_template_line_characters=max(len(r['text']) for r in template_lines))))


if __name__ == '__main__':
    run()
