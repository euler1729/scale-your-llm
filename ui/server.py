#!/usr/bin/env python3
"""
Minimal chat UI server for the Scale-Your-LLM workshop.

Serves a ChatGPT/Claude-like web UI on port 8000 and proxies streaming
chat-completion requests to a llama.cpp OpenAI-compatible server.

No history is persisted; the browser keeps the conversation in memory only.

Env vars:
  PORT          UI port                  (default: 3000)
  LLAMA_SERVER  backend base URL         (default: http://localhost:8000)
  MODEL         model name sent upstream (default: "local-model")
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PORT = int(os.environ.get("PORT", "3000"))
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "http://localhost:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "local-model")

INDEX = (HERE / "index.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[ui] {self.address_string()} - {fmt % args}")

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX, "text/html; charset=utf-8")
        elif self.path == "/api/model":
            self._send(200, json.dumps({"model": MODEL}).encode(),
                       "application/json")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, b"not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b"bad json")
            return

        messages = payload.get("messages", [])
        upstream = {
            "model": MODEL,
            "messages": messages,
            "stream": True,
            "temperature": payload.get("temperature", 0.7),
            "max_tokens": payload.get("max_tokens", 1024),
        }

        # Stream the upstream SSE response straight back to the browser.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        url = f"{LLAMA_SERVER}/v1/chat/completions"
        try:
            with httpx.stream("POST", url, json=upstream, timeout=None) as r:
                if r.status_code != 200:
                    body = r.read().decode(errors="replace")
                    self._chunk(f"data: {json.dumps({'choices':[{'delta':{'content':f'[backend {r.status_code}] {body[:300]}'}}]})}\n\n")
                    return
                for raw in r.iter_lines():
                    if raw:
                        self._chunk(raw + "\n\n")
        except httpx.HTTPError as e:
            msg = f"[cannot reach backend at {url}: {e}]"
            self._chunk(f"data: {json.dumps({'choices':[{'delta':{'content':msg}}]})}\n\n")
        finally:
            try:
                self._chunk("data: [DONE]\n\n")
            except Exception:
                pass

    def _chunk(self, text):
        self.wfile.write(text.encode())
        self.wfile.flush()


def main():
    print(f"[ui] serving on http://localhost:{PORT}")
    print(f"[ui] proxying to {LLAMA_SERVER}  (model={MODEL})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
