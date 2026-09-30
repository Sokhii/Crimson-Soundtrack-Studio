"""Local LLM runtime.

``InferenceBackend`` is the only interface the semantic and matching layers
depend on. Implementations:

* ``LlamaServerBackend`` - launches the bundled ``llama-server`` (llama.cpp)
  from ``runtime/llama/`` on 127.0.0.1 and talks to its OpenAI-compatible API.
  The process is owned by the Studio and stopped with it; no external AI
  application is needed.
* ``ScriptedBackend`` - deterministic replies for tests.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..app_paths import AppPaths
from .catalog import LocalModel, ModelError


class BackendError(ModelError):
    title = "Local AI problem"


class ModelOutputError(BackendError):
    """The model answered, but the answer was unusable (e.g. cut off at the token limit). Retryable."""


class InferenceBackend:
    name = "abstract"
    model_id = ""

    def start(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def chat(self, messages: List[Dict[str, str]], *, json_schema: Optional[dict] = None, max_tokens: int = 1024,
             temperature: float = 0.2) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model_id}


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never proxy localhost traffic


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _opener().open(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:800]
        if exc.code == 500 and "does not match" in body:
            # llama-server rejects output that stopped mid-structure (usually the token limit)
            raise ModelOutputError("The local AI's answer was incomplete.", details=f"HTTP {exc.code}: {body}") from exc
        raise BackendError("The local AI returned an error.", details=f"HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise BackendError("The local AI did not respond.", details=str(exc)) from exc


def runtime_executable_name() -> str:
    return "llama-server.exe" if os.name == "nt" else "llama-server"


def find_llama_server(paths: AppPaths, override: str = "") -> Optional[Path]:
    if override:
        candidate = paths.from_stored(override) or Path(override)
        return candidate if candidate.is_file() else None
    exe = runtime_executable_name()
    for candidate in (paths.runtime / "llama" / exe, paths.runtime / exe):
        if candidate.is_file():
            return candidate
    return next(iter(sorted(paths.runtime.glob(f"**/{exe}"))), None) if paths.runtime.is_dir() else None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaServerBackend(InferenceBackend):
    name = "llama.cpp"

    def __init__(self, paths: AppPaths, model: LocalModel, *, server_path: Optional[Path] = None,
                 gpu_layers: str = "auto", context_size: int = 0, threads: int = 0,
                 request_timeout: float = 600.0, log: Optional[Callable[[str], None]] = None) -> None:
        self.paths = paths
        self.model = model
        self.model_id = model.id
        self.server_path = server_path or find_llama_server(paths)
        self.gpu_layers = str(gpu_layers or "auto")
        self.context_size = context_size or model.context_size or 8192
        self.threads = threads
        self.request_timeout = request_timeout
        self.log = log or (lambda _m: None)
        self.port = 0
        self.base_url = ""
        self.process: Optional[subprocess.Popen] = None
        self.log_file = paths.logs / "llama-server.log"
        self.active_args: List[str] = []
        self.schema_supported = True
        self._lock = threading.Lock()

    def _attempts(self) -> Iterable[List[str]]:
        base = ["-m", str(self.model.install_path(self.paths)), "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.context_size), "--jinja", "--no-webui"]
        if self.threads:
            base += ["-t", str(self.threads)]
        if self.gpu_layers == "auto":
            yield base + ["-ngl", "999"]
            yield base + ["-ngl", "20"]
            yield base + ["-ngl", "0"]
        else:
            yield base + ["-ngl", self.gpu_layers]
            if self.gpu_layers != "0":
                yield base + ["-ngl", "0"]

    def start(self, timeout: float = 600.0) -> None:
        with self._lock:
            if self.process and self.process.poll() is None and self._healthy():
                return
            if self.server_path is None or not self.server_path.is_file():
                raise BackendError("The built-in AI runtime (llama.cpp) was not found.",
                                   hint="Reinstall the portable build; the runtime lives in the 'runtime\\llama' folder.")
            model_path = self.model.install_path(self.paths)
            if not model_path.is_file():
                raise BackendError(f"The model file for {self.model.display_name} is missing.",
                                   hint="Download the model again on the AI Model page.", details=str(model_path))
            errors = []
            for args in self._attempts():
                for drop_webui in (False, True):
                    self.port = _free_port()
                    self.base_url = f"http://127.0.0.1:{self.port}"
                    attempt = [a for a in args if not (drop_webui and a == "--no-webui")]
                    attempt[attempt.index("--port") + 1] = str(self.port)
                    ok, message = self._launch(attempt, timeout)
                    if ok:
                        self.active_args = attempt
                        return
                    errors.append(message)
                    if "no-webui" not in message and "unknown" not in message.lower() \
                            and "invalid argument" not in message.lower():
                        break
            raise BackendError("The local AI model could not be started.",
                               hint="The model may be too large for this computer. Try a lower tier.",
                               details="\n".join(errors[-3:]))

    def _launch(self, args: List[str], timeout: float) -> tuple:
        self.stop()
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["LLAMA_CACHE"] = str(self.paths.cache / "llama")  # any llama.cpp cache stays inside the app folder
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        cmd = [str(self.server_path)] + args
        self.log("starting: " + " ".join(cmd))
        log_handle = open(self.log_file, "ab")
        log_handle.write(("\n==== " + time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(cmd) + "\n").encode())
        log_handle.flush()
        try:
            self.process = subprocess.Popen(cmd, cwd=str(self.server_path.parent), stdout=log_handle,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                            creationflags=creationflags)
        except OSError as exc:
            log_handle.close()
            return False, f"could not execute {self.server_path}: {exc}"
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    return False, f"exited with code {self.process.returncode}: {self._log_tail()}"
                if self._healthy():
                    return True, ""
                time.sleep(0.3)
        finally:
            log_handle.close()
        self.stop()
        return False, "timed out waiting for the model to load"

    def _log_tail(self, lines: int = 12) -> str:
        try:
            return "\n".join(self.log_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-lines:])
        except OSError:
            return ""

    def _healthy(self) -> bool:
        try:
            with _opener().open(self.base_url + "/health", timeout=2) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def chat(self, messages, *, json_schema=None, max_tokens=1024, temperature=0.2) -> str:
        if not self.process or self.process.poll() is not None:
            self.start()
        payload: Dict[str, Any] = {"model": self.model.id, "messages": messages, "max_tokens": max_tokens,
                                   "temperature": temperature, "stream": False, "cache_prompt": True,
                                   # hybrid "thinking" models (e.g. Qwen3) answer directly
                                   "chat_template_kwargs": {"enable_thinking": False}}
        if json_schema is not None and self.schema_supported:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "result", "schema": json_schema, "strict": True}}
        try:
            reply = _post_json(self.base_url + "/v1/chat/completions", payload, self.request_timeout)
        except BackendError as exc:
            if json_schema is not None and self.schema_supported and "response_format" in exc.details.lower():
                self.schema_supported = False
                payload.pop("response_format", None)
                reply = _post_json(self.base_url + "/v1/chat/completions", payload, self.request_timeout)
            else:
                raise
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError("The local AI gave an unexpected reply.", details=str(reply)[:300]) from exc
        content = message.get("content") or ""
        if not content.strip() and message.get("reasoning_content"):
            content = message["reasoning_content"]
        return content

    def stop(self) -> None:
        proc = self.process
        self.process = None
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "server": str(self.server_path), "model": self.model.id,
                "args": self.active_args, "log": str(self.log_file)}


class ScriptedBackend(InferenceBackend):
    """Replays responses (strings or callables receiving the message list). For tests."""

    name = "scripted"

    def __init__(self, responses: List[Any], model_id: str = "scripted-test") -> None:
        self.responses = list(responses)
        self.calls: List[List[Dict[str, str]]] = []
        self.model_id = model_id
        self.started = False

    def start(self) -> None:
        self.started = True

    def chat(self, messages, *, json_schema=None, max_tokens=1024, temperature=0.2) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise BackendError("scripted backend exhausted")
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return item(messages) if callable(item) else item


def run_inference_check(backend: InferenceBackend) -> Dict[str, Any]:
    """Load check: one small schema-constrained request."""

    started = time.monotonic()
    try:
        reply = backend.chat(
            [{"role": "system", "content": "You are a terse assistant."},
             {"role": "user", "content": 'Answer with JSON {"ok": true, "word": "<any word>"}.'}],
            json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "word": {"type": "string"}},
                         "required": ["ok", "word"], "additionalProperties": False},
            max_tokens=128, temperature=0.0)
    except ModelOutputError as exc:
        return {"ok": False, "reply": exc.details[:200], "seconds": round(time.monotonic() - started, 2)}
    try:
        parsed = json.loads(reply)
        ok = isinstance(parsed, dict) and "ok" in parsed
    except ValueError:
        ok = False
    return {"ok": ok, "reply": reply[:200], "seconds": round(time.monotonic() - started, 2)}
