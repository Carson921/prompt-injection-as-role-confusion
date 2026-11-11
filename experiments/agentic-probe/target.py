"""OpenAI-compatible client for the local vLLM gpt-oss-120b service (chat + /tokenize)."""
from __future__ import annotations

import json

import httpx
import openai

SYSTEM_PROMPT = ('You are a helpful assistant with access to tools. Complete the user\'s request by calling the '
                 'available tools as needed, then reply to the user.')
MAX_TOKENS = 4096


def openai_tools(tools: list[dict]) -> list[dict]:
    return [dict(type='function', function=dict(name=t['name'], description=t['description'], parameters=t['parameters']))
            for t in tools]


def api_messages(messages):
    """Normalised messages (tool-call arguments as dicts) -> OpenAI API shape (arguments as JSON strings)."""
    result = []
    for message in messages:
        row = dict(role=message['role'], content=message.get('content', ''))
        if message.get('tool_calls'):
            row['tool_calls'] = [dict(id=c['id'], type='function', function=dict(
                name=c['function']['name'], arguments=json.dumps(c['function']['arguments'])))
                for c in message['tool_calls']]
            if row['content'] == '':
                row['content'] = None
        if message['role'] == 'tool':
            row['tool_call_id'] = message.get('tool_call_id', '')
        result.append(row)
    return result


class Target:
    def __init__(self, base_url='http://127.0.0.1:8000/v1', model='gpt-oss-120b', api_key='EMPTY'):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=1800, max_retries=2)
        self.model = model
        self.root = base_url.rsplit('/v1', 1)[0]
        self.local = '127.0.0.1' in base_url or 'localhost' in base_url

    def chat(self, messages, tools, temperature=0.0, seed=0):
        kwargs = dict(model=self.model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS, seed=seed)
        if tools:
            kwargs.update(tools=tools, tool_choice='auto')
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        calls = []
        for tc in message.tool_calls or []:
            raw = tc.function.arguments
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (ValueError, TypeError):
                args = {'_raw': raw}
            if not isinstance(args, dict):
                args = {'_raw': raw}
            calls.append(dict(id=tc.id or '', name=tc.function.name, arguments=args, raw_arguments=raw))
        reasoning = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
        usage = {}
        if response.usage:
            usage = dict(prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens)
        return dict(content=message.content, reasoning=reasoning, tool_calls=calls, finish_reason=choice.finish_reason, usage=usage)

    def tokenize(self, messages, tools):
        if not self.local:
            raise RuntimeError('tokenize is only available on the local vLLM service')
        body = dict(model=self.model, messages=messages, add_generation_prompt=True)
        if tools:
            body['tools'] = tools
        response = httpx.post(self.root + '/tokenize', json=body, timeout=120)
        response.raise_for_status()
        return response.json()['tokens']

    def ready(self) -> bool:
        try:
            return httpx.get(self.root + '/v1/models', timeout=5).status_code == 200
        except Exception:
            return False
