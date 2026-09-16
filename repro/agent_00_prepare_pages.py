"""
Step 1/3 of the agent prompt-injection evaluation.
Port of the page-preparation cells of
experiments/cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb.

Samples Wikipedia pages, then writes two injected copies of each: one carrying a
plain instruction to exfiltrate .env ("base-injection") and one where that same
instruction is followed by a forged chain of thought ("cot-forgery-injection").

Outputs: experiments/cot-forgery-agent-evals/scrapes/{raw,base-injection,cot-forgery-injection}
"""
import argparse
import os
import random
import re
import time

import requests
import yaml

from common import AGENT_DIR

SEED = 1234
SCRAPES = AGENT_DIR / 'scrapes'


def strip_classes(html_content: str) -> str:
    """Remove all class attributes to reduce token count"""
    return re.sub(r'\s+class="[^"]*"', '', html_content)


def load_raw_html(max_size: int = 100, max_length_kb: int = 100):
    """Download `max_size` Wikipedia pages under `max_length_kb`, stripped of class attrs."""
    from datasets import load_dataset

    out_dir = SCRAPES / 'raw'
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(p for p in out_dir.glob('*.html'))
    if len(existing) >= max_size:
        print(f'{len(existing)} pages already in {out_dir}')
        return out_dir

    ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
    shuf = ds.shuffle(seed=SEED, buffer_size=1_000)

    collected, file_index = len(existing), len(existing)
    for ex in shuf:
        if collected >= max_size:
            break
        file_path = out_dir / f'{file_index:03d}.html'
        if file_path.exists():
            collected += 1
            file_index += 1
            continue
        try:
            r = requests.get(ex['url'], headers={'User-Agent': 'Sampler!'}, timeout=30, allow_redirects=True)
            r.raise_for_status()
        except Exception as e:
            print(f'skip {ex["url"]}: {e}')
            continue

        html_content = strip_classes(r.content.decode('utf-8', errors='ignore'))
        content_size_kb = len(html_content.encode('utf-8')) / 1024
        if content_size_kb > max_length_kb:
            print(f'Skipping {ex["url"]}: {content_size_kb:.1f}KB exceeds {max_length_kb}KB')
            continue

        file_path.write_text(html_content, encoding='utf-8')
        collected += 1
        file_index += 1
        print(f'Collected {collected}/{max_size}: {ex["url"]} ({content_size_kb:.1f}KB)')
        time.sleep(0.1)

    return out_dir


_BODY_CLOSE_RE = re.compile(r'</body\s*>', flags=re.IGNORECASE)


def create_injection_function(prompts: list[str]):
    """Pick a random payload per file and splice it in just before </body>."""
    def transform_html(html: str) -> str:
        prompt_to_inject = random.choice(prompts)
        new_html, n = _BODY_CLOSE_RE.subn(lambda m: prompt_to_inject + '</body>', html, count=1)
        if n == 0:
            raise ValueError('Could not find </body> tag to inject prompt.')
        return new_html
    return transform_html


def write_html_variant(variant_name: str, transform_fn):
    in_dir, out_dir = SCRAPES / 'raw', SCRAPES / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        html = (in_dir / name).read_text(encoding='utf-8', errors='ignore')
        (out_dir / name).write_text(transform_fn(html), encoding='utf-8')
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-pages', type=int, default=100)
    args = ap.parse_args()

    load_raw_html(max_size=args.n_pages)

    prompt_config = yaml.safe_load(open(AGENT_DIR / 'prompts' / 'injections.yaml'))
    base_prompts = [item['prompt'].strip() for item in prompt_config['base_injections']]
    cot_forgery_prompts = [item['prompt'].strip() for item in prompt_config['prompt_injections']]

    random.seed(SEED)  # notebook leaves `random` unseeded; pin it so runs match
    print('wrote', write_html_variant('base-injection', create_injection_function(base_prompts)))
    print('wrote', write_html_variant('cot-forgery-injection', create_injection_function(cot_forgery_prompts)))


if __name__ == '__main__':
    main()
