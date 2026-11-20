"""Quick check of the vLLM DeepSeek service: plain turn, tool call, tool result follow-up, throughput."""
import json, time
from concurrent.futures import ThreadPoolExecutor
from ds_common import Codec
from ds_serving import Vllm
codec, client = Codec(), Vllm()
tools = [dict(type='function', function=dict(name='get_weather', description='Get the weather of a city.',
         parameters=dict(type='object', properties=dict(city=dict(type='string')), required=['city'])))]
messages = [dict(role='system', content='You are a helpful assistant.'), dict(role='user', content='What is the weather in Paris? Use the tool.')]
enc = codec.encode(messages, tools)
t = time.time(); r = client.complete(enc['input_ids'], codec.eos_id); print(round(time.time() - t, 2), 's', r['finish_reason'], r['usage'])
print(repr(r['text']))
parsed = codec.parse(r['text'] + codec.eos); print(parsed)
messages += [dict(role='assistant', content=parsed['content'], reasoning_content=parsed['reasoning_content'], tool_calls=[dict(c, id='c1') for c in parsed['tool_calls']]),
             dict(role='tool', tool_call_id='c1', content='{"city": "Paris", "temp_c": 11, "sky": "rain"}')]
enc = codec.encode(messages, tools)
r = client.complete(enc['input_ids'], codec.eos_id); print(repr(r['text']))
long = codec.encode([dict(role='user', content='Write a 300-word story about a lighthouse.')])
t = time.time()
with ThreadPoolExecutor(24) as pool:
    outs = list(pool.map(lambda i: client.complete(long['input_ids'], codec.eos_id, max_tokens=600), range(24)))
dt = time.time() - t; toks = sum(o['usage']['completion_tokens'] for o in outs)
print('24 concurrent:', round(dt, 1), 's', toks, 'tokens', round(toks / dt, 1), 'tok/s')
