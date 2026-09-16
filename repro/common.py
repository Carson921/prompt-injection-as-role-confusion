"""
Shared helpers for the OpenRouter-only reproduction of
"Prompt Injection as Role Confusion" (arXiv:2603.12277).

All LLM traffic goes through OpenRouter. The only target model is
openai/gpt-oss-20b; google/gemini-2.5-pro is kept for the two auxiliary
roles the paper uses it for (CoT Forgery generation and harm/injection
classification).
"""
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / '.env')
sys.path.insert(0, str(REPO))

from utils.openrouter import send_async_requests  # noqa: E402  (needs env + path first)

# ---------------------------------------------------------------- config ----

# Paper used nebius/fp4; Nebius no longer serves gpt-oss-20b on OpenRouter, so
# we pin the closest equivalent (fp4 quant, supports `tools` + `reasoning`).
TARGET_MODEL = {'model': 'openai/gpt-oss-20b', 'provider': 'parasail/fp4'}

# Auxiliary model (forgery author + LLM judge), as in the paper.
AUX_MODEL = {'model': 'google/gemini-2.5-pro', 'provider': 'google-ai-studio'}

NO_SAMPLE_PARAMS = {
    'temperature': 0, 'top_p': 1, 'topk_k': 1,
    'frequency_penalty': 0, 'presence_penalty': 0, 'repetition_penalty': 1,
}

CHAT_DIR = REPO / 'experiments' / 'cot-forgery-chat-evals'
AGENT_DIR = REPO / 'experiments' / 'cot-forgery-agent-evals'
CACHE_DIR = REPO / 'repro' / 'cache'


def model_params(spec: dict, **extra) -> dict:
    """Build an OpenRouter request body (minus `messages`) pinned to one provider."""
    return {
        'model': spec['model'],
        'provider': {'order': [spec['provider']], 'allow_fallbacks': False},
        'usage': {'include': True},  # makes OpenRouter report per-call cost
        **extra,
    }


# ------------------------------------------------------------- requests ----

class OpenRouterError(RuntimeError):
    pass


class RateLimiter:
    """Global request pacer. OpenRouter caps new accounts at 20 rpm on gemini-2.5-pro."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.lock = asyncio.Lock()
        self.next_slot = 0.0

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if wait:
            await asyncio.sleep(wait)


_LIMITER: RateLimiter | None = None


async def _post(session, payload):
    """POST one chat completion, raising on anything that should be retried."""
    if _LIMITER is not None:
        await _LIMITER.acquire()
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']}
    timeout = aiohttp.ClientTimeout(total=420, sock_connect=30)
    async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            raise OpenRouterError(f'HTTP {resp.status}: {(await resp.text())[:300]}')
        body = await resp.json()
    if 'choices' not in body:
        raise OpenRouterError(f'no choices: {json.dumps(body)[:300]}')
    return body


def _prompt_hash(prompt, params) -> str:
    blob = json.dumps({'p': prompt, 'm': params.get('model')}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


async def run_openrouter(prompts: list[list[dict]], params: dict, cache_name: str,
                         batch_size: int = 20, max_retries: int = 6,
                         requests_per_minute: int | None = None) -> list[dict]:
    """
    Send `prompts` (each a message list) to OpenRouter and return the raw responses.

    Responses are cached to disk chunk by chunk, each tagged with a hash of the
    prompt that produced it, so an interrupted run resumes where it stopped and a
    changed prompt set invalidates the entries that no longer apply. Unlike the
    repo helper, non-200 responses raise so the retry/backoff loop sees rate limits.
    """
    global _LIMITER
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f'{cache_name}.json'
    want = [_prompt_hash(p, params) for p in prompts]

    done: list[dict] = []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        # Keep the longest prefix whose prompts still match.
        for i, rec in enumerate(cached):
            if i >= len(want) or rec.get('h') != want[i]:
                break
            done.append(rec)
        if len(done) < len(cached):
            print(f'[cache] {cache_path.name}: dropped {len(cached) - len(done)} entries '
                  f'that no longer match the current prompts')
        if len(done) >= len(prompts):
            print(f'[cache] reusing {len(prompts)} responses from {cache_path.name}')
            return [r['r'] for r in done[:len(prompts)]]
        if done:
            print(f'[cache] resuming {cache_path.name} at {len(done)}/{len(prompts)}')

    if requests_per_minute is None:
        requests_per_minute = 18 if 'gemini' in params.get('model', '') else 120
    _LIMITER = RateLimiter(requests_per_minute)

    start = len(done)
    remaining = prompts[start:]
    chunks = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
    for n, chunk in enumerate(chunks, 1):
        responses = await send_async_requests(
            inputs=[{'messages': p, **params} for p in chunk],
            request_generator=_post,
            batch_size=len(chunk),
            max_retries=max_retries,
            verbose=False,
        )
        offset = start + sum(len(c) for c in chunks[:n - 1])
        done.extend({'h': want[offset + j], 'r': r} for j, r in enumerate(responses))
        cache_path.write_text(json.dumps(done))
        spent = sum((rec['r'].get('usage') or {}).get('cost', 0) or 0 for rec in done)
        print(f'  [{cache_name}] {len(done)}/{len(prompts)} (chunk {n}/{len(chunks)}, ${spent:.2f})', flush=True)

    _LIMITER = None
    return [r['r'] for r in done]


def extract(llm_response: dict) -> dict:
    """Pull (reasoning, output) out of a response, dropping length-truncated ones."""
    if 'choices' not in llm_response:
        return {'reasoning': None, 'output': None}
    choice = llm_response['choices'][0]
    if choice.get('finish_reason') == 'length':
        return {'reasoning': None, 'output': None}
    msg = choice.get('message', {})
    return {'reasoning': msg.get('reasoning'), 'output': msg.get('content')}


def usage_summary(responses: list[dict]) -> str:
    prompt_toks = sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in responses)
    completion_toks = sum((r.get('usage') or {}).get('completion_tokens', 0) for r in responses)
    cost = sum((r.get('usage') or {}).get('cost', 0) or 0 for r in responses)
    return f'{len(responses)} calls | {prompt_toks:,} in / {completion_toks:,} out tokens | ${cost:.3f}'


def arun(coro):
    return asyncio.run(coro)
