"""Thin client for the vLLM /v1/completions endpoint, driven with token ids from ds_common.Codec."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ds_common import GENERATION


class Vllm:
    def __init__(self, base_url='http://127.0.0.1:8000/v1', model='qwen35-397b', max_model_len=65536, timeout=3600):
        self.base_url, self.model, self.max_model_len, self.timeout = base_url.rstrip('/'), model, max_model_len, timeout

    def _post(self, path, payload):
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        last = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                body = error.read().decode(errors='replace')[:500]
                if error.code < 500:
                    raise RuntimeError(f'vLLM HTTP {error.code}: {body}') from error
                last = RuntimeError(f'vLLM HTTP {error.code}: {body}')
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last = error
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f'vLLM transport failure: {last}')

    def complete(self, input_ids, eos_id, max_tokens=None):
        """Greedy completion of exactly these token ids; returns text with special tokens kept."""
        budget = min(max_tokens or GENERATION['max_tokens'], self.max_model_len - len(input_ids) - 8)
        if budget < 256:
            raise RuntimeError(f'Prompt of {len(input_ids)} tokens leaves no generation budget')
        result = self._post('/completions', dict(
            model=self.model, prompt=list(map(int, input_ids)), max_tokens=budget, temperature=GENERATION['temperature'],
            seed=GENERATION['seed'], skip_special_tokens=False, spaces_between_special_tokens=False,
            add_special_tokens=False, stop_token_ids=[int(eos_id)]))
        choice = result['choices'][0]
        return dict(text=choice['text'], finish_reason=choice.get('finish_reason'), usage=result.get('usage'),
                    max_tokens=budget)

    def tokenize(self, text):
        return self._post('/../tokenize', dict(model=self.model, prompt=text, add_special_tokens=False))['tokens']

    def ready(self):
        try:
            with urllib.request.urlopen(self.base_url + '/models', timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
