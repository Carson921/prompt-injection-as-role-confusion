"""Shared helpers for the benign-goal synthetic prompt-injection dataset (arXiv 2611.11861, Sec. 3).

Paths, JSON helpers, the 53-domain taxonomy (Table 6), a small OpenRouter client with a
call log and cost accounting, and prompt rendering / payload location that mirror
experiments/ipi-arena-richevo/common.py so the frozen role probe can score the exact
prompt the vLLM service consumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IPI_RICHEVO = ROOT / 'experiments/ipi-arena-richevo'
MODEL = Path('/tmp/wz55191-agentdojo-first-tool-model')
sys.path.insert(0, str(HERE))

SPECIAL = {'<|start|>': 200006, '<|end|>': 200007, '<|message|>': 200008, '<|channel|>': 200005,
           '<|return|>': 200002, '<|call|>': 200012, '<|constrain|>': 200003}

DOMAINS = {
    'Information & Data': ['weather', 'news', 'stock_market', 'sports_scores', 'encyclopedia', 'dictionary',
                           'translation', 'unit_conversion', 'currency_exchange'],
    'Communication': ['email', 'messaging', 'calendar', 'contacts', 'notifications'],
    'Entertainment': ['music', 'movies', 'books', 'games', 'recipes'],
    'Travel & Location': ['flights', 'hotels', 'restaurants', 'maps', 'ride_sharing'],
    'Productivity': ['task_management', 'note_taking', 'file_management', 'spreadsheets'],
    'E-commerce': ['shopping', 'order_tracking', 'price_comparison', 'product_reviews'],
    'Health & Fitness': ['health_tracking', 'medication_reminders', 'workout_planner'],
    'Smart Home': ['lighting', 'thermostat', 'security_cameras', 'appliances'],
    'Developer Tools': ['code_execution', 'database_queries', 'api_monitoring', 'log_analysis'],
    'Finance': ['banking', 'payments', 'invoicing', 'expense_tracking'],
    'Social Media': ['social_posting', 'analytics', 'content_moderation'],
    'Education': ['tutoring', 'quiz_generation', 'course_management'],
}
DOMAIN_CATEGORY = {d: c for c, ds in DOMAINS.items() for d in ds}


def sha(value) -> str:
    if isinstance(value, Path):
        value = value.read_bytes()
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def openrouter_key(path=Path.home() / '.bashrc') -> str:
    for line in path.read_text().splitlines():
        match = re.match(r'\s*export\s+(OPENROUTER_API_KEY|OPENROUTER_KEY)\s*=\s*["\']?([^"\'\s]+)', line)
        if match:
            return match.group(2)
    raise ValueError('OPENROUTER_KEY not found in ~/.bashrc')


def parse_json(text: str):
    """Parse a JSON object/array from an LLM reply, tolerating code fences and prose."""
    text = text.strip()
    fence = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                continue
    raise ValueError('No JSON found in reply: ' + text[:200])


class LLM:
    """OpenRouter chat client with retries, a JSONL call log and cost accounting."""

    def __init__(self, log_path: Path, model: str = 'google/gemini-2.5-flash', max_usd: float = 20.0, temperature: float = 0.7):
        import httpx
        self.key, self.model, self.max_usd, self.temperature = openrouter_key(), model, max_usd, temperature
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(base_url='https://openrouter.ai/api/v1', timeout=180,
                                   headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        self.lock = threading.Lock()
        self.spent, self.calls = 0.0, 0

    def chat(self, system: str, user: str, *, json_mode: bool = True, temperature: float | None = None,
             max_tokens: int = 4096, retries: int = 4, model: str | None = None) -> str:
        with self.lock:
            if self.spent >= self.max_usd:
                raise RuntimeError(f'OpenRouter budget exhausted: ${self.spent:.2f}')
        body = dict(model=model or self.model, temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens, usage={'include': True},
                    messages=[dict(role='system', content=system), dict(role='user', content=user)])
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        error = None
        for attempt in range(retries):
            started = time.time()
            try:
                response = self.client.post('/chat/completions', json=body)
                data = response.json()
                if response.status_code != 200 or 'choices' not in data or not data['choices']:
                    raise RuntimeError(f'HTTP {response.status_code}: {json.dumps(data)[:300]}')
                content = data['choices'][0]['message'].get('content') or ''
                cost = float((data.get('usage') or {}).get('cost') or 0.0)
                with self.lock:
                    self.spent += cost
                    self.calls += 1
                    with self.log_path.open('a') as stream:
                        stream.write(json.dumps(dict(unix=started, model=body['model'], seconds=time.time() - started,
                                                     cost=cost, usage=data.get('usage'), system=system[:200],
                                                     user=user[:4000], content=content), ensure_ascii=False) + '\n')
                return content
            except Exception as exc:  # network / provider error: back off and retry
                error = f'{type(exc).__name__}: {exc}'
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f'OpenRouter call failed after {retries} attempts: {error}')

    def chat_json(self, system: str, user: str, **kwargs):
        last = None
        for _ in range(3):
            try:
                return parse_json(self.chat(system, user, **kwargs))
            except ValueError as exc:
                last = exc
        raise last


def render_prompt(tokenizer, messages: list[dict], tools: list[dict], date: str) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                         add_generation_prompt=True, reasoning_effort='medium',
                                         strftime_now=lambda fmt: date)


def locate_payload(prompt: str, offsets, payload: str, goal: str) -> list[dict]:
    """Every rendered occurrence of the (JSON-escaped) payload and the goal tokens inside it.

    The payload is placed inside a JSON string field of the tool response, so it appears
    JSON-escaped once; the chat template may serialise the tool content again, so the
    twice-escaped rendering is searched too.
    """
    once_payload, once_goal = json.dumps(payload, ensure_ascii=False)[1:-1], json.dumps(goal, ensure_ascii=False)[1:-1]
    variants = [(json.dumps(once_payload, ensure_ascii=False)[1:-1], json.dumps(once_goal, ensure_ascii=False)[1:-1]),
                (once_payload, once_goal), (payload, goal)]
    occupied, occurrences = [], []
    for rendered_payload, rendered_goal in dict.fromkeys(variants):
        relative = rendered_payload.find(rendered_goal)
        if relative < 0:
            raise ValueError('Escaped goal missing from escaped payload')
        position = 0
        while True:
            start = prompt.find(rendered_payload, position)
            if start < 0:
                break
            end = start + len(rendered_payload)
            position = end
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            goal_start, goal_end = start + relative, start + relative + len(rendered_goal)
            indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
            goal_indices = [i for i in indices if offsets[i][0] < goal_end and offsets[i][1] > goal_start]
            occurrences.append(dict(start_char=start, end_char=end, goal_start_char=goal_start, goal_end_char=goal_end,
                                    token_indices=indices, goal_token_indices=goal_indices))
    occurrences.sort(key=lambda row: row['start_char'])
    if not occurrences:
        raise ValueError('Payload not exposed in rendered prompt')
    return occurrences


def special_floor(model_path: Path) -> tuple[int, list[str]]:
    source = json.loads((model_path / 'tokenizer.json').read_text())
    floor = max(source['model']['vocab'].values()) + 1
    spellings = [row['content'] for row in source['added_tokens'] if row['special']]
    return floor, spellings
