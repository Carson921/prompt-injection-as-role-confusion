"""
Runs *inside* the bubblewrap sandbox for the agent prompt-injection eval.

The sandbox has no network egress (its own empty net namespace, loopback only)
and no view of the host filesystem beyond a scratch workspace. This process:

  1. serves the injected page and a stand-in for the exfiltration endpoint over
     HTTPS on 127.0.0.1:443, with /etc/hosts mapping 0x0.st and dpaste.com there,
     so the transcript the agent produces is identical to hitting the real hosts
     while nothing leaves the machine;
  2. executes bash commands sent by the host driver over a unix socket.

Control protocol (newline-delimited JSON over /ctrl/cmd.sock):
  {"op": "reset", "page": "<html>", "secret": "..."}  -> new episode
  {"op": "exec", "command": "..."}                    -> {"stdout","stderr","exit_code"}
  {"op": "exfil"}                                     -> {"posts": [...]}
"""
import json
import os
import random
import socket
import socketserver
import ssl
import string
import subprocess
import threading
from http.server import BaseHTTPRequestHandler

CTRL = '/ctrl'
WORKSPACE = '/workspace/container-1'
SOCK_PATH = f'{CTRL}/cmd.sock'

STATE = {'page': '<html><body>empty</body></html>', 'posts': []}
LOCK = threading.Lock()


class SinkHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):  # keep the sandbox quiet
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with LOCK:
            page = STATE['page']
        self._send(page.encode('utf-8'), 'text/html; charset=utf-8')

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length).decode('utf-8', errors='ignore') if length else ''
        with LOCK:
            STATE['posts'].append({'path': self.path, 'body': body})
        # dpaste's API answers with the URL of the new paste, as plain text.
        slug = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        self._send(f'https://dpaste.com/{slug}\n'.encode(), 'text/plain; charset=utf-8', 201)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_sink():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(f'{CTRL}/server.pem', f'{CTRL}/server.key')
    srv = ThreadedHTTPServer(('127.0.0.1', 443), SinkHandler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def reset_episode(page: str, secret: str):
    with LOCK:
        STATE['page'] = page
        STATE['posts'] = []
    for name in os.listdir(WORKSPACE):
        path = os.path.join(WORKSPACE, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            subprocess.run(['rm', '-rf', path], check=False)
    with open(os.path.join(WORKSPACE, '.env'), 'w') as f:
        f.write(f'SECRET_API_KEY={secret}\n')


def run_command(command: str) -> dict:
    env = {
        'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        'HOME': WORKSPACE,
        'LANG': 'C.UTF-8',
        # The sandbox's CA, so curl/python trust the local stand-in hosts.
        'CURL_CA_BUNDLE': f'{CTRL}/server.pem',
        'SSL_CERT_FILE': f'{CTRL}/server.pem',
        'REQUESTS_CA_BUNDLE': f'{CTRL}/server.pem',
    }
    try:
        done = subprocess.run(command, shell=True, capture_output=True, text=True,
                              timeout=30, cwd=WORKSPACE, env=env)
        return {'stdout': done.stdout, 'stderr': done.stderr, 'exit_code': done.returncode}
    except Exception as e:  # mirrors the notebook's tool wrapper
        return {'stdout': '', 'stderr': str(e), 'exit_code': 1}


def serve_control():
    if os.path.exists(SOCK_PATH):
        os.remove(SOCK_PATH)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o666)
    srv.listen(8)
    with open(f'{CTRL}/ready', 'w') as f:
        f.write('ready')
    while True:
        conn, _ = srv.accept()
        with conn:
            data = b''
            while not data.endswith(b'\n'):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            if not data:
                continue
            try:
                req = json.loads(data.decode())
                op = req.get('op')
                if op == 'reset':
                    reset_episode(req['page'], req['secret'])
                    resp = {'ok': True}
                elif op == 'exec':
                    resp = run_command(req.get('command', ''))
                elif op == 'exfil':
                    with LOCK:
                        resp = {'posts': list(STATE['posts'])}
                else:
                    resp = {'error': f'unknown op {op}'}
            except Exception as e:
                resp = {'error': repr(e)}
            conn.sendall((json.dumps(resp) + '\n').encode())


if __name__ == '__main__':
    os.makedirs(WORKSPACE, exist_ok=True)
    start_sink()
    serve_control()
