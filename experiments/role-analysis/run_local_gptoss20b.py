"""Execute locally adapted copies of the four original role-analysis notebooks."""
import argparse
import json
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'outputs/local-gptoss20b'


def environment():
    for key, value in {
        'HF_HOME':'/root/.cache/huggingface-role-probe',
        'HF_HUB_CACHE':'/root/.cache/huggingface-role-probe/hub',
        'HF_DATASETS_CACHE':'/root/.cache/huggingface-role-probe/datasets',
        'UV_CACHE_DIR':'/root/.cache/uv-role-probe',
        'UV_LINK_MODE':'copy', 'OMP_NUM_THREADS':'8',
        'CUDA_VISIBLE_DEVICES':'0', 'PYTHONUNBUFFERED':'1',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
    }.items():
        os.environ[key] = value
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH','')
    for folder in ['data/conversations', 'outputs/probes', 'outputs/probe-training',
                   'outputs/probe-projections', 'plots', 'outputs/local-gptoss20b/plots']:
        (HERE/folder).mkdir(parents=True, exist_ok=True)


def original(name):
    nb = nbformat.read(HERE/name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
        cell.source = cell.source.replace('/workspace/deliberative-alignment-jailbreaks', str(ROOT))
    return nb


def conversations():
    nb = original('01-get-conversations-data.ipynb')
    nb.cells[3].source = '# Dataset preparation uses the original sampling functions; cached below.'
    nb.cells[4].source = '# OASST user turns were prepared with the original tree traversal and filters.'
    nb.cells[5].source = "user_queries_df = pd.read_csv('outputs/local-gptoss20b/user_queries.csv')\ndisplay(user_queries_df.groupby('dataset').agg(conversations=('conv_id','nunique')))"
    nb.cells[7].source = "target_models = [{'model':'openai/gpt-oss-20b','model_prefix':'gptoss-20b'}]"
    nb.cells[8].source = "print('OpenRouter generation skipped: using the local checkpoint below.')"
    nb.cells[10].source = nb.cells[10].source.replace('from utils.loader import', 'from local_runtime import')
    nb.cells[11].source = nb.cells[11].source.replace(
        'local_result_df = generate_convs_data_local(',
        'from local_runtime import generate_local_batch\nlocal_result_df = generate_convs_data_local(')
    nb.cells[11].source = nb.cells[11].source.replace('BATCH_SIZE = 16','BATCH_SIZE = 64').replace(
        'return [lst[i:i + n] for i in range(0, len(lst), n)]',
        'return [lst[:16]] + [lst[i:i+n] for i in range(16,len(lst),n)]')
    nb.cells[12].source = """# Parse Harmony channels directly: the original tail(1) discarded all CoT segments.
from local_runtime import parse_latest_response, atomic_json, OUT
raw_df = local_result_df.copy()
pairs = [parse_latest_response(s) if isinstance(s,str) else (None,None) for s in raw_df['state_text']]
raw_df['cot'] = [p[0] for p in pairs]
raw_df['parsed_final'] = [p[1] for p in pairs]
failed = raw_df.loc[raw_df[['assistant','cot','parsed_final']].isna().any(axis=1), 'conv_id'].unique()
print('Dropping incomplete conversations:',len(failed))
"""
    nb.cells[13].source = """merged = (raw_df[~raw_df['conv_id'].isin(failed)]
    .drop(columns=['state_text','assistant']).rename(columns={'parsed_final':'assistant'}))
merged.to_csv(f'{ws}/experiments/role-analysis/data/conversations/{model_prefix}.csv',index=False)
atomic_json(OUT/'conversation_generation.json', {
    'requested_conversations':int(user_queries_df.conv_id.nunique()),
    'requested_user_turns':len(user_queries_df), 'complete_conversations':int(merged.conv_id.nunique()),
    'complete_user_turns':len(merged), 'dropped_incomplete_conversations':len(failed),
    'cross_model_comparison':'skipped by user instruction',
})
display(merged.groupby('dataset').agg(conversations=('conv_id','nunique')))
"""
    return nb


def execute(nb, name, kernel='role-probe-uv'):
    path = OUT / f'{name}.executed.ipynb'
    nb.metadata.kernelspec = {'name':kernel,'display_name':kernel,'language':'R' if kernel.endswith('-r') else 'python'}
    class LiveClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg['header']['msg_type'] == 'stream':
                print(msg['content']['text'],end='',flush=True)
            return super().process_message(msg,cell,cell_index)
    client = LiveClient(nb, timeout=None, kernel_name=kernel,
                            resources={'metadata':{'path':str(HERE)}}, record_timing=True)
    started = time.time()
    with client.setup_kernel():
        for index,cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            if name == '02-train-role-probes' and index == 25:
                while not (OUT/'conversation_generation.json').exists():
                    print('Waiting for local conversation generation before validation.',flush=True)
                    time.sleep(30)
            print(f'{name}: starting original cell {index}',flush=True)
            try:
                client.execute_cell(cell,index)
            finally:
                nbformat.write(nb,path)
            print(f'{name}: completed cell {index}; elapsed {time.time()-started:.1f}s',flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['conversations','training','r-analysis','r-tomato'],required=True)
    args = parser.parse_args()
    environment()
    os.environ['ROLE_ANALYSIS_MODEL_BACKEND'] = 'bf16' if args.stage == 'training' else 'mxfp4'
    if args.stage == 'conversations':
        execute(conversations(),'01-get-conversations-data')
    else:
        from local_notebooks import training, r_analysis, r_tomato
        if args.stage == 'training':
            execute(training(),'02-train-role-probes')
        elif args.stage == 'r-analysis':
            execute(r_analysis(),'03-analyze-probes','role-analysis-r')
        else:
            execute(r_tomato(),'04-tomato-probe-results','role-analysis-r')


if __name__ == '__main__':
    main()
