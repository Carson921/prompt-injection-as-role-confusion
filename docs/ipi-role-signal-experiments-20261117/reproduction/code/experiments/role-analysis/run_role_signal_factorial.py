"""Frozen paired text x native wrapper x visible-context-position experiment.

The checkpoint specified by the user is a six-role 120B multi-head probe.
No probe or backbone training, checkpoint selection, generation, or ASR test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from functools import lru_cache
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = HERE / 'outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz'
REQUESTED = '/home/wz55191/Github/SPAR/prompt-injection-as-role-confusion/experiments/role-analysis/outputs/c4-dolma3-single-agentdojo-mixed-2080ti-20261103/checkpoint.npz'
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
SHA = 'fe602209cab66a433c34978dab8b2e8157f372ca5fb1c7f6881882401a6ffa33'
ROLES = ('system', 'developer', 'user', 'cot', 'assistant', 'tool')
SEED = 20261112
SYSTEM = '<|start|>system<|message|>You are a helpful assistant.<|end|>'
HEADERS = {'cot': '<|start|>assistant<|channel|>analysis<|message|>',
           'user': '<|start|>user<|message|>'}

# Newly authored neutral descriptions; no role declarations, instructions,
# dialogue labels, reasoning prompts, or repeated filler blocks.
FILLERS = {
'A': '''A stone wall borders the garden. A bird rests nearby. The morning light falls across a narrow gravel path. Small leaves lie beside a shallow pool. The water reflects a pale patch of sky between two tall trees. A wooden bench stands under the branches of an old oak. Its surface is smooth in the middle and rough along the edges. Moss covers the lower stones near the shaded corner. Beyond the wall, a meadow slopes toward a line of distant hills.
White flowers grow beside a clay pot near the gate. A bee moves slowly from one blossom to the next. The stems bend slightly in a light breeze. Several brown seed heads remain among the fresh green shoots. The path curves around a bed of low shrubs. Their leaves vary from silver grey to deep green. A small feather rests on the dry ground beneath the bench. Thin clouds drift above the open field. The air carries the scent of damp soil and cut grass.
Farther along the path, a broad tree casts an uneven shadow over the stones. A few roots rise above the ground near its trunk. The bark has deep ridges and patches of soft lichen. A shallow hollow holds a little rainwater. Fallen twigs lie beneath the outer branches. The grass grows shorter in the sunny part of the meadow. A hedge marks the far side of the field. Its upper branches form a dark line against the light sky. There are small gaps between the leaves.
The pool has a smooth rim of flat stones. One corner lies in the shade for most of the morning. A few floating leaves gather near that corner. Clear water reveals sand and rounded pebbles at the bottom. A thin reed rises from the damp edge. The surrounding soil is darker than the path. Tiny plants grow between two stones in the rim. Their narrow leaves spread close to the ground. A passing cloud softens the bright reflection on the surface.
Near the gate, the wall is topped with broad flat slabs. Their surfaces have small pits filled with dust and bits of dry leaf. A trailing plant hangs over one slab. Its newest growth is a lighter shade of green. The gate is made of narrow wooden boards with a simple iron latch. Sunlight catches the upper edge of the metal. A small patch of bare earth lies just inside the entrance. The gravel grows deeper near the bend in the path.
The meadow contains several kinds of grass. Some have fine stems and loose seed heads. Others form thicker clumps close to the ground. Pale flowers appear among the shorter blades. A narrow strip of taller growth follows the hedge. The hills beyond the field look blue in the morning haze. A small group of trees stands near the lowest part of the skyline. Their shapes remain distinct beneath the thin clouds. The light changes gently across the open ground.
Under the oak, dry leaves gather in a shallow dip. Some are curled and brittle, while others are dark and flat. Acorn caps lie among small pieces of bark. A patch of fern grows beside an exposed root. Its delicate leaves catch a little light from the edge of the canopy. The bench legs rest on two broad stones. Grass presses against the outside of one stone. A faint trail crosses the soil between the bench and the pool. The nearby branches move slowly.
At the other end of the garden, a low bed contains herbs with small leaves. Several stems carry pale purple flowers. A clay dish sits on a level patch of ground beside the bed. Its rim is chipped on one side. Sand has collected in the shallow marks on its surface. The soil around the herbs is loose and dry at the top. Darker earth appears beneath a fallen leaf. The path continues along the wall toward a quiet shaded corner. A light breeze passes through the grass.''',
'B': '''A broad window lights the gallery. A vase stands nearby. Pale walls surround a room with a smooth wooden floor. The glass holds a faint reflection of the ceiling. Several low shelves stand along the far wall. A woven basket rests on one shelf beside a shallow ceramic bowl. The bowl has a thin blue line around its rim. A narrow table occupies the middle of the room. Its legs cast soft shadows on the floor. Outside the window, a row of trees borders a quiet courtyard.
The vase has a rounded body and a short neck. Small variations in its surface catch the light. One side appears warmer where the sun reaches it. The nearby basket is woven from thin strips of pale wood. Its handle curves above an empty opening. A folded cloth lies at the edge of the shelf. The fabric has a loose weave and a slightly uneven border. On the opposite wall hangs a painting of a coastal scene. Grey clouds fill much of its sky.
The painting shows a low shore with several dark rocks. A strip of pale sand lies between the rocks and a calm stretch of water. The horizon is soft and slightly blurred. The frame is plain wood with a narrow inner edge. Fine grain runs along its upper surface. A small patch of sunlight falls on the wall below it. The floorboards in this part of the room have a warm brown colour. Their joints form long parallel lines toward the doorway. The doorway opens into a smaller room.
Inside the smaller room, a tall cabinet stands against a cream wall. Its glass doors reveal a group of cups and dishes. Some pieces are white, while others have green or brown glazes. A shallow drawer sits below the glass. The metal handles have darkened around their edges. Beside the cabinet is a chair with a curved back. Its seat is covered with a plain linen cushion. The lower wall remains in shadow. A wide beam crosses the ceiling above the doorway.
In the courtyard, square paving stones surround a small patch of earth. A tree rises from the middle of that patch. Its leaves cast moving shapes across the stones. The trunk leans slightly toward the brighter side of the space. A low ledge runs along one wall. Several dry leaves have gathered below it. The wall has a rough surface with small flecks of darker stone. A narrow band of shade lies near its base. Above the wall, the sky looks clear and pale.
Back in the main room, the table holds a broad wooden tray. A smooth pebble rests beside a small empty glass. Light passes through the glass and forms a bright curve on the tray. The wood has a darker knot near one corner. The table edge is rounded and worn smooth. Beneath it, the floor has a few shallow marks. A long rug lies near the entrance. Its pattern consists of quiet bands of red and brown. The ends of the rug are finished with short threads.
The gallery ceiling is high and gently sloped. Thin beams cross the pale surface at regular intervals. A hanging lamp has a simple white shade. Its cord forms a straight line from the ceiling. The room is bright enough for the lamp to remain dark. Near the window, the wall has a slightly cooler colour. The sill is wide and holds a small pot with a leafy plant. New leaves cluster at the end of one stem. The pot stands on a plain round saucer.
Along the lower shelf, several books lie beside a covered box. Their cloth bindings have faded unevenly. The box is made of dark wood with a flat lid. A tiny brass hinge is visible at the back. The shelf casts a narrow shadow on the wall behind it. The open space below reveals the grain of the floorboards. Beyond the window, a breeze moves the outer branches of the courtyard tree. A cloud passes slowly across the bright sky. Soft light remains on the quiet room.'''
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def jsonlines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dump_lines(path, values):
    Path(path).write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


@lru_cache(maxsize=32)
def filler(tok, name, budget):
    """Deterministic maximal whole-sentence prefix plus natural exact-length ending."""
    shortest = {
        ('A', 13): 'A stone wall borders a garden with a path and a pool.',
        ('A', 15): 'A low stone wall borders a quiet garden with a path and a pool.',
        ('B', 13): 'A broad window lights a gallery with a table and a shelf.',
        ('B', 15): 'A broad window lights a quiet gallery with a wooden table and a shelf.',
    }
    if (name, budget) in shortest:
        text = shortest[name, budget]
        ids = enc(tok, text)
        assert len(ids) == budget
        return text, ids, 0
    sentences = FILLERS[name].replace('\n', ' ').split('. ')
    sentences = [s.strip().rstrip('.') + '.' for s in sentences]
    endings = (['The air is calm.', 'A bird rests nearby.', 'The light is soft.',
                'A pale feather lies beside a smooth stone.', 'The leaves move gently in the warm afternoon air.',
                'A narrow shadow stretches over the dry grass near the garden wall.',
                'The quiet path is lined with small green plants and a few pale flowers.',
                'The ground beneath the old tree is covered with dry leaves and tiny pieces of bark.'] if name == 'A' else
               ['The room is calm.', 'A vase stands nearby.', 'The light is soft.',
                'A small glass rests beside a wooden bowl.', 'The window reflects a patch of pale afternoon sky.',
                'A narrow shadow stretches over the wooden floor near the gallery wall.',
                'The wide shelf holds a small round pot and a few smooth stones.',
                'The space beneath the wide window is filled with soft light and the shadows of nearby leaves.'])
    endings = endings + [a + ' ' + b for i,a in enumerate(endings) for j,b in enumerate(endings) if i != j]
    for count in range(len(sentences), -1, -1):
        prefix = ' '.join(sentences[:count])
        if len(enc(tok, prefix)) > budget:
            continue
        for end in [''] + endings:
            text = prefix + (' ' if prefix and end else '') + end
            ids = enc(tok, text)
            if len(ids) == budget:
                return text, ids, count
    raise ValueError(f'No complete neutral prose of exactly {budget} tokens')


def build_input(tok, pair, style, wrapper, position, filler_id):
    system = enc(tok, SYSTEM)
    pre_header, end = enc(tok, HEADERS['user']), enc(tok, '<|end|>')
    header = enc(tok, HEADERS[wrapper])
    budget = position - len(system) - len(pre_header) - len(end) - len(header)
    filler_text, prefix_ids, whole_sentences = filler(tok, filler_id, budget)
    body = pair[style]
    body_ids = enc(tok, body)
    ids = system + pre_header + prefix_ids + end + header + body_ids + end
    body_start = len(ids) - len(body_ids) - len(end)
    assert body_start == position and ids[body_start:body_start + len(body_ids)] == body_ids
    assert len(ids) < 4096 and all(i < 199998 for i in body_ids + prefix_ids)
    # Native boundaries are actual special IDs, not strings that merely resemble them.
    rendered = tok.decode(ids, skip_special_tokens=False)
    assert enc(tok, rendered) == ids
    return dict(input_id=f'{pair["pair_id"]}-{filler_id}-{style}-{wrapper}-{position}',
                pair_id=pair['pair_id'], split=pair['split'], filler_id=filler_id,
                text_style=style, wrapper=wrapper, position=position,
                input_ids=ids, body_start=body_start, body_end=body_start + len(body_ids),
                body_token_ids=body_ids, body_tokens=len(body_ids),
                header_start=body_start-len(header), header_length=len(header),
                filler_tokens=len(prefix_ids), filler_text=filler_text,
                filler_complete_prefix_sentences=whole_sentences,
                filler_header_adjustment_tokens=5-len(header),
                excluded_boundary_tokens=0, body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                rendered_prompt=rendered)


def prepare(args):
    out = args.output
    assert not (out / 'protocol.json').exists(), 'Use a fresh output directory'
    out.mkdir(parents=True, exist_ok=True)
    assert digest(args.probe) == SHA
    tok = tokenizer(args.model)
    pairs = jsonlines(args.pairs)
    assert Counter(p['split'] for p in pairs) == {'construction': 2, 'pilot': 10, 'formal': 50}
    assert len({p['pair_id'] for p in pairs}) == 62
    dump_lines(out / 'pairs.jsonl', pairs)
    write(out / 'neutral-prefixes.json', FILLERS)
    inputs, construction = [], []
    for pair in pairs:
        for fill in ('A', 'B') if pair['split'] == 'formal' else ('A',):
            for style in ('styled', 'destyled'):
                for wrapper in ('cot', 'user'):
                    for position in (32, 256, 768):
                        row = build_input(tok, pair, style, wrapper, position, fill)
                        (construction if pair['split'] == 'construction' else inputs).append(row)
    rng = np.random.default_rng(SEED)
    inputs = [inputs[i] for i in rng.permutation(len(inputs))]
    dump_lines(out / 'inputs.jsonl', inputs)
    dump_lines(out / 'construction-inputs.jsonl', construction)
    classifier = load_module('_factorial_classifier', HERE / 'synthetic_agent_classifier.py')
    probe = classifier.load_probe(args.probe)
    write(out / 'protocol.json', dict(
        frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), date='2026-11-12',
        model='openai/gpt-oss-120b', model_path=str(args.model), probe_requested=REQUESTED,
        probe_local=str(args.probe.resolve()), probe_sha256=SHA, probe_metadata=probe.metadata,
        adaptation='User-specified checkpoint replaces plan default 20B/layer12 five-role probe with its bound 120B/six-role/16-head interface; no new fitting or selection.',
        roles=ROLES, factors=dict(text_style=['styled','destyled'],wrapper=['cot','user'],position=[32,256,768],filler_id=['A','B']),
        formal_pairs=50, formal_inputs=1200, pilot_pairs=10, pilot_inputs=120,
        construction_pairs=2, construction_inputs=24, positions_zero_based=True,
        system_prefix=SYSTEM, headers=HEADERS, ending='<|end|> for both wrappers',
        native_format='Minimal native Harmony headers assembled explicitly, without apply_chat_template auto-generated long system/developer preamble, to permit position32.',
        filler_method='Deterministic maximal whole-sentence prefix plus a natural exact-length final sentence; all text visible. Header lengths5/3 compensated by2 filler tokens.',
        primary_metric='Mean over body token logits z_cot-z_user, then equal weight across semantic units, conditions, and filler versions.',
        primary_effects=['text styled-destyled','wrapper cot-user','position256-32','position768-32'],
        interactions=['text x wrapper at each position','change in text x wrapper versus32'],
        confidence=dict(method='paired semantic-unit cluster percentile bootstrap',replicates=2000,seed=SEED,level=.95,pointwise=True),
        inference=dict(eval=True,dropout=False,truncation=False,seed=SEED,batch_size=1,attention='tiled original eager',query_tile=256,feature_cast='BF16 -> F16 -> F32'),
        layer_supplement='Not available: this checkpoint is a fixed nonlinear joint multi-layer attention-head readout. No separately matched per-layer probes; never apply it to arbitrary layers.',
        pairs_sha256=digest(out/'pairs.jsonl'),inputs_sha256=digest(out/'inputs.jsonl'),
        analysis_code_sha256=digest(HERE/'analyze_role_signal_factorial.py'),runner_sha256=digest(__file__),
        statistical_selection='n=50 and analysis frozen before any formal readout; no significance tests or sequential expansion. Pilot for engineering and variance diagnostic only.'))
    write(out / 'construction-validation.json', dict(status='passed',construction_inputs=len(construction),
          inputs=len(inputs),all_body_starts_exact=True,full_decode_reencode_equal=True,
          boundary_excluded=0,body_special_tokens=0,pilot_formal_source_disjoint=True,
          body_token_range=[min(r['body_tokens'] for r in inputs),max(r['body_tokens'] for r in inputs)]))
    print(json.dumps(dict(stage='prepared',inputs=len(inputs),construction=len(construction))),flush=True)


def normal_checks(tok):
    bodies = [
      'A small reading room has two wide windows and a wooden table. Three blue books lie beside an empty glass. A lamp stands near the wall, and a green plant grows in a round clay pot. Sunlight reaches the floor below the window. The room is quiet during the afternoon.',
      'The morning train reaches the station at nine. A clock hangs above the entrance, and several benches stand along the platform. A brown bag rests beside a folded coat. Beyond the tracks, low buildings face a line of trees. Thin clouds cover part of the sky.']
    rows=[]
    for i,body in enumerate(bodies):
        for role in ROLES:
            header = HEADERS['cot'] if role=='cot' else f'<|start|>{role}<|message|>'
            if role=='assistant':
                header='<|start|>assistant<|channel|>final<|message|>'
            # Normal system/developer precede user; assistant/tool receive preceding user context.
            prefix = '' if role=='system' else SYSTEM
            if role in ('cot','assistant','tool'):
                prefix += '<|start|>user<|message|>Describe the scene.<|end|>'
            content = body
            if role=='tool':
                prefix += '<|start|>assistant to=functions.scene<|channel|>commentary json<|message|>{}<|call|>'
                header='<|start|>functions.scene to=assistant<|channel|>commentary<|message|>'
                content=json.dumps(body,ensure_ascii=False)
            head = enc(tok,prefix+header); b=enc(tok,content)
            rows.append(dict(input_id=f'neutral-{i}-{role}',role=role,input_ids=head+b+enc(tok,'<|end|>'),
                             body_start=len(head),body_end=len(head)+len(b)))
    return rows


def run(args):
    import torch
    import pandas as pd
    out=args.output
    protocol=json.loads((out/'protocol.json').read_text())
    assert digest(args.probe)==protocol['probe_sha256']==SHA
    assert digest(out/'inputs.jsonl')==protocol['inputs_sha256']
    assert digest(out/'pairs.jsonl')==protocol['pairs_sha256']
    assert digest(HERE/'analyze_role_signal_factorial.py')==protocol['analysis_code_sha256']
    scorer=load_module('_factorial_scorer',ROOT/'experiments/autodan-role-probe/scorer.py')
    collect,spans,attention,classifier=scorer.dependencies()
    probe=classifier.load_probe(args.probe); binding=probe.metadata['model_binding']
    assert tuple(probe.metadata['roles'])==ROLES
    for name,expected in binding['assets_sha256'].items():
        assert digest(args.model/name)==expected, name
    proof=json.loads((args.model/'staging-proof.json').read_text())
    assert proof['all_shards_verified'] and {r['name']:r['sha256'] for r in proof['shards']}==binding['weight_shards_sha256']
    os.environ['HF_HOME']=str(ROOT/'experiments/single-message-pre-mlp/outputs/huggingface')
    os.environ['TRITON_CACHE_DIR']='/tmp/wz55191-role-signal-triton'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    torch.set_num_threads(2); torch.manual_seed(SEED)
    net=probe.torch(device='cpu').eval()
    mean,scale=torch.from_numpy(probe.mean),torch.from_numpy(probe.scale)
    tok=tokenizer(args.model)
    started=time.monotonic(); model,device_map=collect.load_model(args.model)
    attention.install_tiled_attention(256)
    write(out/'runtime.json',dict(device_map=device_map,hardware=[torch.cuda.get_device_name(i) for i in range(2)],
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','kernels','triton','numpy','pandas','pyarrow','matplotlib']},
        torch_version=torch.__version__,python=platform.python_version(),model_load_seconds=time.monotonic()-started,
        staging_proof_sha256=digest(args.model/'staging-proof.json')))

    def forward(row, folder):
        destination=out/folder/(row['input_id']+'.npz')
        if destination.exists():
            with np.load(destination,allow_pickle=False) as z:
                return {k:z[k] for k in z.files}
        positions=np.arange(row['body_start'],row['body_end'],dtype=np.int64)
        native,_=collect.forward_capture(model,probe.metadata['heads'],row['input_ids'],positions)
        assert native.dtype==torch.bfloat16
        features=native.to(torch.float16).reshape(len(positions),1024).numpy()
        with torch.inference_mode():
            logits=net((torch.from_numpy(features).float()-mean)/scale).numpy()
            probs=torch.from_numpy(logits).softmax(1).numpy()
        result=dict(positions=positions,features=features,logits=logits,probabilities=probs,roles=np.asarray(ROLES))
        assert np.isfinite(logits).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=destination.with_suffix('.tmp.npz'); np.savez_compressed(temp,**result);temp.replace(destination)
        return result

    normals=normal_checks(tok)
    (out/'normal-checks').mkdir(exist_ok=True)
    dump_lines(out/'normal-checks/inputs.jsonl',normals)
    checks=[]
    for row in normals:
        z=forward(row,'normal-checks')
        p=z['probabilities']; checks.append(dict(input_id=row['input_id'],role=row['role'],body_tokens=len(p),
           native_role_mean_probability=float(p[:,ROLES.index(row['role'])].mean()),
           token_accuracy=float((p.argmax(1)==ROLES.index(row['role'])).mean()),
           **{'p_'+r:float(p[:,j].mean()) for j,r in enumerate(ROLES)}))
    write(out/'normal-checks/summary.json',dict(checks=checks,diagnostic_only=True,no_selection=True))
    rows=jsonlines(out/'inputs.jsonl')
    rows=sorted(rows,key=lambda r:r['split']!='pilot') # each split retains frozen random order
    summaries=[]; phase_start=time.monotonic();pilot_start=phase_start
    for i,row in enumerate(rows):
        t=time.monotonic();z=forward(row,'readouts')
        logits,p=z['logits'].astype(float),z['probabilities'].astype(float)
        summaries.append({k:row[k] for k in ['input_id','pair_id','split','filler_id','text_style','wrapper','position','body_tokens']} |
                         dict(mean_margin=float((logits[:,3]-logits[:,2]).mean()),
                              **{'p_'+r:float(p[:,j].mean()) for j,r in enumerate(ROLES)},
                              **{'z_'+r:float(logits[:,j].mean()) for j,r in enumerate(ROLES)},
                              argmax_cot_fraction=float((p.argmax(1)==3).mean()),seconds=time.monotonic()-t))
        if (i+1)%12==0 or i+1==len(rows):
            pd.DataFrame(summaries).to_csv(out/'input-summary.csv',index=False)
            status=dict(stage='forward',done=i+1,total=len(rows),split=row['split'],seconds=time.monotonic()-phase_start)
            write(out/'status.json',status); print(json.dumps(status),flush=True)
        if i+1==120:
            pilot=pd.DataFrame(summaries)
            diff=pilot.pivot(index=['pair_id','wrapper','position'],columns='text_style',values='mean_margin')
            per_pair=(diff.styled-diff.destyled).groupby('pair_id').mean()
            write(out/'pilot-summary.json',dict(inputs=120,pairs=10,seconds=time.monotonic()-pilot_start,
                paired_text_effect_sd=float(per_pair.std(ddof=1)),formal_n_fixed=50,
                approximate_formal_seconds=(time.monotonic()-pilot_start)*10,
                rule='Engineering pilot only; pre-frozen n=50 retained regardless of pilot effects.'))
    # Exact repeated single-input forwards and causality of post-body ending.
    validation=[]
    for row in [next(r for r in rows if r['split']=='formal' and r['position']==p) for p in (32,256,768)]:
        z=forward(row,'readouts');positions=np.arange(row['body_start'],row['body_end'])
        native,_=collect.forward_capture(model,probe.metadata['heads'],row['input_ids'],positions)
        replay=native.to(torch.float16).reshape(-1,1024).numpy()
        short,_=collect.forward_capture(model,probe.metadata['heads'],row['input_ids'][:row['body_end']],positions)
        causal=short.to(torch.float16).reshape(-1,1024).numpy()
        validation.append(dict(input_id=row['input_id'],repeat_max_abs=float(np.max(abs(replay-z['features']))),
                               ending_removal_max_abs=float(np.max(abs(causal-z['features'])))))
    write(out/'forward-validation.json',dict(batch_size=1,batch_single_comparison='Not applicable: every input forwarded alone without padding.',checks=validation))
    write(out/'run-manifest.json',dict(status='complete',inputs=len(rows),formal_inputs=1200,pilot_inputs=120,
        normal_inputs=12,seconds=time.monotonic()-started,probe_sha256=SHA,inputs_sha256=digest(out/'inputs.jsonl'),
        runner_sha256=digest(__file__),summary_sha256=digest(out/'input-summary.csv'),
        feature_files=len(list((out/'readouts').glob('*.npz'))),all_readouts_complete=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,default=HERE/'outputs/role-signal-factorial-20261112')
    p.add_argument('--pairs',type=Path,default=HERE/'role_signal_pairs.jsonl')
    p.add_argument('--probe',type=Path,default=PROBE)
    p.add_argument('--model',type=Path,default=MODEL)
    a=p.parse_args(); (prepare if a.action=='prepare' else run)(a)


if __name__=='__main__':
    main()
