"""Persistent native Llama 4 Scout probe server: load the four-H100 backbone and the frozen probe once,
then serve hot-reloadable operations (ds_probe_ops.py) over local HTTP so experiments and the
optimizer can restart without the long model load.

POST /op   {"op": "<function in ds_probe_ops>", "args": {...}}  ->  {"ok": true, "result": ...}
GET  /health
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent


class State:
    def __init__(self, output):
        from ds_scorer import Prober, prepare_native_runtime
        prepare_native_runtime()
        started = time.time()
        self.prober = Prober(output)
        self.model, self.probe, self.heads, self.backbone = self.prober.model, self.prober.probe, self.prober.heads, self.prober.backbone
        self.load_seconds = time.time() - started
        self.caches, self.ops, self.ops_mtime = {}, None, None

    def operations(self):
        path = HERE / 'ds_probe_ops.py'
        mtime = path.stat().st_mtime
        if self.ops is None:
            self.ops = importlib.import_module('ds_probe_ops')
        elif mtime != self.ops_mtime:
            self.ops = importlib.reload(self.ops)
        self.ops_mtime = mtime
        return self.ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = State(args.output)
    print(json.dumps(dict(stage='ready', load_seconds=state.load_seconds, port=args.port)), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(dict(ok=True, load_seconds=state.load_seconds, caches=len(state.caches), pid=os.getpid()))

        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                started = time.time()
                result = getattr(state.operations(), request['op'])(state, **request.get('args', {}))
                self.reply(dict(ok=True, result=result, seconds=time.time() - started))
            except Exception as error:
                self.reply(dict(ok=False, error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()[-3000:]), 500)

    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
