"""Independently parse serialized role labels and audit completed feature caches."""
import json

import numpy as np
import pandas as pd

import systemness_data as data


def main():
    tok=data.runtime.tokenizer()
    special={name:tok.convert_tokens_to_ids(name) for name in ['<|start|>','<|message|>','<|end|>','<|return|>','<|call|>']}
    counts={}
    for name,source,metadata in [('main',data.CACHE,data.OUT/'data-complete.json'),
                                 ('supplement',data.CACHE/'supplement',data.OUT/'supplement-data-complete.json')]:
        meta=json.loads(metadata.read_text())
        assert all(data.digest(source/file)==sha for file,sha in meta['cache_sha256'].items())
        rows=pd.read_parquet(source/'rows.parquet')
        tokens=json.loads((source/'token-ids.json').read_text())
        assert rows.sample_ix.is_unique
        assert rows[rows.wrapper=='no-tags'].role.isna().all()
        checked=0
        for prompt,group in rows[rows.wrapper!='no-tags'].groupby('prompt_ix',sort=False):
            header=[];in_header=False;active=None;actual={}
            for i,token in enumerate(tokens[prompt]):
                if token==special['<|start|>']:
                    in_header=True;header=[];active=None
                elif token==special['<|message|>']:
                    text=tok.decode(header);in_header=False
                    active=('tool' if text.startswith('functions.') else
                            'tool_call' if 'to=functions.' in text else
                            'cot' if text.startswith('assistant') and 'analysis' in text else
                            'assistant' if text.startswith('assistant') else
                            'system' if text.startswith('system') else
                            'user' if text.startswith('user') else None)
                elif token in [special['<|end|>'],special['<|return|>'],special['<|call|>']]:
                    active=None
                elif in_header:
                    header.append(token)
                elif active:
                    actual[i]=active
            assert all(actual.get(int(row.token_ix))==row.role for row in group.itertuples()),(name,prompt)
            checked+=len(group)
        counts[name]=dict(rows=len(rows),formal_rows_independently_parsed=checked,all_labels_match=True)
    keys=json.loads((data.OUT/'document-keys.json').read_text())
    assert len(keys)==len({key['body_sha256'] for key in keys})==1800
    for first,second in [('fit','validation'),('fit','test'),('validation','test')]:
        for field in ['body_sha256','full_body_sha256','source_url']:
            a={key[field] for key in keys if key['split']==first and key.get(field)}
            b={key[field] for key in keys if key['split']==second and key.get(field)}
            assert not a&b,(first,second,field)
    features={}
    for part in ['development','augmentation','test','challenge']:
        metadata=data.OUT/f'extraction-{part}.json'
        if not metadata.exists():continue
        meta=json.loads(metadata.read_text())
        folder=data.FEATURE_CACHE/part
        assert data.digest(folder/'rows.parquet')==meta['rows_sha256']
        matrix=np.memmap(folder/'features.f16',mode='r',dtype=np.float16,shape=(meta['rows'],8640))
        for start in range(0,len(matrix),4096):
            block=matrix[start:start+4096]
            assert np.isfinite(block).all() and (block!=0).any(axis=1).all(),part
        assert data.digest(folder/'features.f16')==meta['features_sha256']
        features[part]=dict(rows=meta['rows'],all_finite=True,no_unwritten_zero_rows=True,sha256_matches=True)
    report=dict(labels=counts,features=features,body_full_body_and_url_splits_disjoint=True,
                system_is_normal_class=True,source_sha256=data.digest(__file__))
    data.atomic_json(data.OUT/'independent-data-validation.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
