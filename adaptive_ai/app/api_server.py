"""OpenAI-совместимый API-сервер для интеграции с Open WebUI и Cherry Studio.

Запуск: python -m adaptive_ai.app.api_server
Или через launcher.py

Эндпоинты:
  GET  /health           — проверка здоровья
  GET  /v1/models        — список моделей
  POST /v1/chat/completions — чат-комплишн (streaming и non-streaming)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# Устанавливаем UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adaptive_ai.app.config import Config
from adaptive_ai.core.contracts import Status


class ApiHandler(BaseHTTPRequestHandler):
    """Обработчик OpenAI-совместимого API."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        self.config = Config.load()
        self.config.ensure_dirs()
        # Инициализация компонентов
        self._init_components()
        super().__init__(*args, **kwargs)

    def _init_components(self):
        try:
            from adaptive_ai.app.main_impl import _FallbackBackend
            from adaptive_ai.models.model_registry import load_backend
            import adaptive_ai.models.rwkv_backend  # noqa: F401

            backend = load_backend(self.config.model_backend, self.config.model_path)
            if not backend.is_loaded():
                backend = _FallbackBackend()
                backend.load(None)
            self.backend = backend
        except Exception as e:
            from adaptive_ai.app.main_impl import _FallbackBackend

            self.backend = _FallbackBackend()
            self.backend.load(None)

        try:
            from adaptive_ai.memory.sqlite_store import SqliteMemoryStore

            self.memory = SqliteMemoryStore(self.config.db_path)
        except Exception:
            self.memory = None

        from adaptive_ai.tools.registry import ToolRegistry
        self.tools = ToolRegistry()

    def _send_json(self, data: dict[str, Any], status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, data: str):
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "model": self.backend.name})
        elif self.path == "/v1/models":
            self._handle_models()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_models(self):
        models = [
            {
                "id": self.backend.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "adaptive-ai",
            }
        ]
        self._send_json({"object": "list", "data": models})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_chat_completions(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        messages = data.get("messages", [])
        stream = data.get("stream", False)
        max_tokens = data.get("max_tokens", self.config.max_tokens)
        temperature = data.get("temperature", self.config.temperature)
        top_p = data.get("top_p", self.config.top_p)

        # Формируем промпт из messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)
        if not prompt.endswith("Assistant:"):
            prompt += "\n\nAssistant:"

        model_id = data.get("model", self.backend.name)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        if stream:
            self._handle_streaming(prompt, completion_id, model_id, max_tokens, temperature, top_p)
        else:
            self._handle_non_streaming(prompt, completion_id, model_id, max_tokens, temperature, top_p)

    def _handle_non_streaming(self, prompt, completion_id, model_id, max_tokens, temperature, top_p):
        result = self.backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        )

        if result.status != Status.OK:
            self._send_json({"error": result.error or "Generation failed"}, 500)
            return

        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.data or "",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len((result.data or "").split()),
                "total_tokens": len(prompt.split()) + len((result.data or "").split()),
            },
        }
        self._send_json(response)

    def _handle_streaming(self, prompt, completion_id, model_id, max_tokens, temperature, top_p):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Начальный chunk
        initial = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        self._send_sse(json.dumps(initial, ensure_ascii=False))

        full_text = ""
        for chunk in self.backend.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty,
        ):
            if chunk.status == Status.OK and chunk.data:
                full_text += chunk.data
                sse_data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.data},
                            "finish_reason": None,
                        }
                    ],
                }
                self._send_sse(json.dumps(sse_data, ensure_ascii=False))

        # Финальный chunk
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        self._send_sse(json.dumps(final, ensure_ascii=False))
        self._send_sse("[DONE]")

    def log_message(self, format, *args):
        # Минимальное логирование
        pass


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Запустить API-сервер."""
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"Adaptive Intelligence API server running on http://{host}:{port}")
    print(f"OpenAI-compatible endpoint: http://{host}:{port}/v1")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Intelligence API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()
    run_server(args.host, args.port)
