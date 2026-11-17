"""Tool-calling attacker client on OpenRouter with cost accounting and a JSONL call log."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import openai

from common import openrouter_key
from ratelimit import limiter


class AttackerClient:
    def __init__(self, model: str, log_path: Path, max_usd: float = 25.0, temperature: float = 1.0, reasoning_effort: str | None = None, rpm: float = 50.0):
        self.client = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=openrouter_key(), timeout=600, max_retries=2)
        self.model, self.max_usd, self.temperature, self.reasoning_effort = model, max_usd, temperature, reasoning_effort
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.spent, self.calls, self.prompt_tokens, self.completion_tokens = 0.0, 0, 0, 0
        self.limiter = limiter(f'attacker:{model}', rpm)

    def budget_left(self) -> bool:
        with self.lock:
            return self.spent < self.max_usd

    def chat(self, messages: list[dict], tools: list[dict], agent: str = '') -> dict:
        if not self.budget_left():
            raise RuntimeError(f'attacker budget exhausted: ${self.spent:.2f}')
        kwargs = dict(model=self.model, messages=messages, tools=tools, tool_choice='auto', temperature=self.temperature,
                      max_tokens=4096, extra_body={'usage': {'include': True}})
        if self.reasoning_effort:
            kwargs['extra_body']['reasoning'] = {'effort': self.reasoning_effort}
        error = None
        for attempt in range(8):
            started = time.time()
            self.limiter.acquire()
            try:
                response = self.client.chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError('no choices: ' + json.dumps(response.model_dump())[:300])
                message = response.choices[0].message
                calls = []
                for tc in message.tool_calls or []:
                    try:
                        args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
                    except ValueError:
                        args = {'_raw': tc.function.arguments}
                    calls.append(dict(id=tc.id or f'call_{len(calls)}', name=tc.function.name, arguments=args if isinstance(args, dict) else {'_raw': args}))
                usage = response.usage
                cost = float(getattr(usage, 'cost', None) or (usage.model_extra or {}).get('cost', 0.0) or 0.0) if usage else 0.0
                with self.lock:
                    self.spent += cost
                    self.calls += 1
                    if usage:
                        self.prompt_tokens += usage.prompt_tokens or 0
                        self.completion_tokens += usage.completion_tokens or 0
                    with self.log_path.open('a') as stream:
                        stream.write(json.dumps(dict(unix=started, agent=agent, model=self.model, seconds=time.time() - started, cost=cost,
                                                     prompt_tokens=usage.prompt_tokens if usage else None,
                                                     completion_tokens=usage.completion_tokens if usage else None,
                                                     content=(message.content or '')[:2000], tool_calls=calls), ensure_ascii=False) + '\n')
                return dict(content=message.content or '', tool_calls=calls, finish_reason=response.choices[0].finish_reason)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                time.sleep(min(60, 8 * (attempt + 1)))
        raise RuntimeError(f'attacker call failed: {error}')

    def status(self) -> dict:
        with self.lock:
            return dict(model=self.model, spent_usd=round(self.spent, 4), calls=self.calls,
                        prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens)
