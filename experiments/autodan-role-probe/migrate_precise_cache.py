"""Correct cloned auxiliary payload origins using identical native token captures.

Never changes source artifacts or any goal-token row, probability or fitness.
Only rows outside immutable goal can be removed from the captured payload.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from transformers import AutoTokenizer
from precise_scorer import PreciseScorer
from scorer import dependencies, sha, write, ROLES

HERE = Path(__file__).resolve().parent

def migrate(output):
    output = Path(output).resolve()
    manifest_path = output / 'native_cache_import.json'
    manifest = json.loads(manifest_path.read_text())
    assert not manifest.get('auxiliary_payload_origins_corrected')
    dataset = json.loads((output / 'dataset.json').read_text())
    scorer = object.__new__(PreciseScorer)
    scorer.protocol = json.loads((output / 'score_protocol.json').read_text())
    scorer.tokenizer = AutoTokenizer.from_pretrained('/tmp/wz55191-agentdojo-first-tool-model', local_files_only=True)
    scorer.spans = dependencies()[1]
    scorer.native_special_id_floor = 199998
    scorer.native_special_spellings = scorer.protocol['payload_delimiter_policy']['reject_real_special_spellings']
    scorer.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072))
    corrections = []
    for entry in manifest['records']:
        path = Path(entry['destination_score_path'])
        result = json.loads(path.read_text())
        changed = False
        for context, record in zip(dataset['contexts'], result['per_case']):
            request, prompt, ids, offsets, positions, goal, location, input_sha = scorer.prepare(context, result['template'])
            artifact = Path(record['token_artifact'])
            with np.load(artifact, allow_pickle=False) as raw:
                arrays = {key: raw[key] for key in raw.files}
            assert np.array_equal(arrays['input_ids'], ids)
            assert np.array_equal(arrays['input_token_offsets'], offsets)
            old_positions, old_goal = arrays['payload_token_indices'], arrays['goal_mask']
            assert np.array_equal(old_positions[old_goal], positions[goal]), 'Immutable goal token set changed'
            if (np.array_equal(old_positions, positions)
                    and np.array_equal(arrays['payload_char_spans'], [x['payload_char_span'] for x in location['occurrences']])
                    and np.array_equal(arrays['goal_char_spans'], [x['goal_char_span'] for x in location['occurrences']])):
                continue
            mapping = {int(value): index for index, value in enumerate(old_positions)}
            selected = np.array([mapping[int(value)] for value in positions], dtype=np.int64)
            assert np.array_equal(old_goal[selected], goal)
            original_sha = {name: sha(artifact.parent/name) for name in ('tokens.npz','request.json')}
            for key in ('native_bf16_bits','features_fp16','probabilities'):
                before_goal = arrays[key][old_goal].copy()
                arrays[key] = arrays[key][selected]
                assert np.array_equal(before_goal, arrays[key][goal]), key
            arrays.update(payload_token_indices=positions, goal_mask=goal, template_mask=~goal,
                          payload_char_spans=np.asarray([x['payload_char_span'] for x in location['occurrences']]),
                          goal_char_spans=np.asarray([x['goal_char_span'] for x in location['occurrences']]))
            arrays.update({key: np.asarray(value) for key,value in location.items() if key.endswith('_char_span')})
            temporary = artifact.with_name('tokens.corrected.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(artifact)  # replace hard link; source inode remains untouched
            write(artifact.parent/'request.json', {**request,'rendered_prompt':prompt,'location':location,
                  'input_sha256':input_sha,'input_tokens':len(ids),'case_id':context['case_id']})
            probabilities = arrays['probabilities']
            means = {name:dict(zip(ROLES,map(float,probabilities[mask].mean(0,dtype=np.float64))))
                     for name,mask in [('goal',goal),('template',~goal),('full_payload',np.ones(len(goal),bool))]}
            assert all(means['goal'][role] == record['probabilities']['goal'][role] for role in ROLES)
            record.update(probabilities=means,region_means=means,location=location,
                          template_tokens=int((~goal).sum()),goal_tokens=int(goal.sum()))
            record['artifact_sha256'] = {name:sha(artifact.parent/name) for name in ('tokens.npz','request.json')}
            derivation = dict(case_id=context['case_id'],source_npz_sha256=original_sha['tokens.npz'],
                              source_request_sha256=original_sha['request.json'],
                              source_npz_rows_for_precise_payload=selected.tolist(),
                              removed_non_goal_rows=len(old_positions)-len(positions),
                              complete_goal_token_indices_bitwise_identical=True,
                              complete_goal_native_bf16_features_fp16_probabilities_bitwise_identical=True,
                              complete_goal_role_means_max_delta=0.,model_forwards_recomputed=0)
            record['cache_derivation'] = derivation
            write(artifact.parent/'score.json',record)
            corrections.append(dict(template_sha256=entry['template_sha256'],**derivation))
            changed = True
        if changed:
            result['cache_derivation'] = 'Only auxiliary payload start/rows corrected; all goal rows and fitness identical'
            write(path,result)
        entry['destination_score_sha256'] = sha(path)
        assert sha(Path(entry['source_score_path'])) == entry['source_score_sha256']
    manifest.update(auxiliary_payload_origins_corrected=True,
                    preserved_raw_npz_and_request_bytes=(not corrections),
                    only_paths_rewritten=(not corrections),
                    complete_goal_rows_and_fitness_unchanged=True,
                    correction_cases=len(corrections),corrections=corrections,
                    migration_source_sha256=sha(Path(__file__)))
    write(manifest_path,manifest)
    print(json.dumps(dict(candidates=len(manifest['records']),corrected_cases=len(corrections),
                          removed_non_goal_token_rows=sum(x['removed_non_goal_rows'] for x in corrections),
                          model_forwards_recomputed=0,goal_probability_change=0.)),flush=True)

if __name__=='__main__':
    import sys
    migrate(sys.argv[1])
