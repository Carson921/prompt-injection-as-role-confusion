"""Cache the original notebook's datasets, without loading a GPU model."""
import json
from pathlib import Path

import nbformat

from run_local_gptoss20b import environment, HERE, OUT


def main():
    environment()
    import pandas as pd
    from datasets import load_dataset
    from IPython.display import display
    import yaml
    conversations = nbformat.read(HERE/'01-get-conversations-data.ipynb',as_version=4)
    namespace = {'seed':1234,'load_dataset':load_dataset,'pd':pd,'display':display}
    for index in [3,4,5]:
        exec(conversations.cells[index].source,namespace)
    namespace['user_queries_df'].to_csv(OUT/'user_queries.csv',index=False)
    training = nbformat.read(HERE/'02-train-role-probes.ipynb',as_version=4)
    source = training.cells[8].source.replace("open('config/probe.yaml')",f"open({str(HERE/'config/probe.yaml')!r})")
    namespace.update(seed=123,yaml=yaml,model_prefix='gptoss-20b')
    exec(source,namespace)
    (OUT/'raw_training_data.json').write_text(json.dumps(namespace['raw_data'],ensure_ascii=False))
    print('Cached conversation user turns:',len(namespace['user_queries_df']))
    print('Cached raw training texts:',len(namespace['raw_data']))


if __name__ == '__main__':
    main()
