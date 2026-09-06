"""Single-session Codex bridge for an isolated ALFWorld episode."""

from __future__ import annotations

import json
import secrets
import socketserver
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


MAX_REQUEST_BYTES = 16 * 1024
MAX_ACTION_CHARS = 512


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


@dataclass
class AlfworldExecEpisode:
    """Own one host-side environment and expose only observe/step operations."""

    env_manager: Any
    observation: dict[str, Any]
    info: dict[str, Any]
    max_steps: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    environment_done: bool = False
    won: bool = False
    limit_reached: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _admissible_actions(self) -> list[str]:
        commands = self.info.get("admissible_commands")
        if commands is None:
            pools = getattr(
                getattr(self.env_manager, "envs", None), "get_admissible_commands", None
            )
            if pools:
                commands = pools[0]
        if commands is None:
            return []
        return [str(command) for command in commands if str(command) != "help"]

    @property
    def done(self) -> bool:
        return self.environment_done or self.limit_reached

    def _state(self, *, reward: float = 0.0, action_valid: bool | None = None) -> dict[str, Any]:
        anchor = self.observation.get("anchor") or [""]
        state: dict[str, Any] = {
            "ok": True,
            "step": len(self.steps),
            "max_steps": self.max_steps,
            "done": self.done,
            "environment_done": self.environment_done,
            "step_limit_reached": self.limit_reached,
            "won": self.won,
            "reward": float(reward),
            "observation": str(anchor[0]),
            "admissible_actions": self._admissible_actions(),
        }
        if action_valid is not None:
            state["action_valid"] = action_valid
        return state

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return self._state()

    def step(self, action: Any) -> dict[str, Any]:
        if not isinstance(action, str):
            raise ValueError("action must be a string")
        action = action.strip()
        if not action:
            raise ValueError("action cannot be empty")
        if len(action) > MAX_ACTION_CHARS:
            raise ValueError(f"action exceeds {MAX_ACTION_CHARS} characters")
        if any(char in action for char in ("\x00", "\r", "\n", "<", ">")):
            raise ValueError("action contains forbidden control or tag characters")

        with self._lock:
            if self.done:
                state = self._state()
                state["error"] = "episode is already complete"
                return state

            model_response = (
                "<think>Selected through the isolated ALFWorld tool.</think>"
                f"<action>{action}</action>"
            )
            next_observation, rewards, dones, infos = self.env_manager.step([model_response])
            reward = float(rewards[0])
            info = infos[0] if infos else {}
            environment_done = _as_bool(dones[0])
            won = bool(info.get("won", False)) if environment_done else False
            action_valid = _as_bool(info.get("is_action_valid", True))

            self.observation = next_observation
            self.info = info
            self.environment_done = environment_done
            self.won = won
            self.steps.append(
                {
                    "step": len(self.steps),
                    "action": action,
                    "reasoning": None,
                    "model_response": model_response,
                    "env_feedback": str((next_observation.get("anchor") or [""])[0]),
                    "reward": reward,
                    "done": environment_done,
                    "action_valid": action_valid,
                }
            )
            if len(self.steps) >= self.max_steps and not self.environment_done:
                self.limit_reached = True
            return self._state(reward=reward, action_valid=action_valid)

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("operation") or "").strip().lower()
        if operation in {"observe", "status"}:
            return self.observe()
        if operation == "step":
            return self.step(request.get("action"))
        raise ValueError("operation must be observe, status, or step")


class _EpisodeServer(socketserver.TCPServer):
    allow_reuse_address = False

    def __init__(self, controller: AlfworldExecEpisode, token: str):
        self.controller = controller
        self.token = token
        super().__init__(("127.0.0.1", 0), _EpisodeRequestHandler)


class _EpisodeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(10)
        raw = self.rfile.readline(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            self._write({"ok": False, "error": "request is too large"})
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            supplied = str(request.pop("token", ""))
            if not secrets.compare_digest(supplied, self.server.token):
                raise PermissionError("invalid episode token")
            response = self.server.controller.dispatch(request)
        except Exception as exc:  # noqa: BLE001
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._write(response)

    def _write(self, response: dict[str, Any]) -> None:
        payload = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(payload)
        self.wfile.flush()


@contextmanager
def serve_episode(controller: AlfworldExecEpisode) -> Iterator[tuple[int, str]]:
    """Serve one controller for exactly the lifetime of one Codex invocation."""
    token = secrets.token_urlsafe(32)
    server = _EpisodeServer(controller, token)
    thread = threading.Thread(
        target=server.serve_forever,
        name="alfworld-codex-bridge",
        daemon=True,
    )
    thread.start()
    try:
        yield int(server.server_address[1]), token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def build_client_script(*, port: int, token: str) -> str:
    """Return a stdlib-only client copied into the Codex workspace."""
    return f'''#!/opt/cobras/bin/python
import json
import socket
import sys

HOST = "127.0.0.1"
PORT = {int(port)}
TOKEN = {token!r}


def request(operation, action=None):
    payload = {{"token": TOKEN, "operation": operation}}
    if action is not None:
        payload["action"] = action
    with socket.create_connection((HOST, PORT), timeout=10) as sock:
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\\n")
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\\n" in chunk:
                break
    response = json.loads(b"".join(chunks).decode("utf-8"))
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("ok") else 2


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {{"observe", "status", "step"}}:
        print("usage: python alfworld_tool.py observe|status|step ACTION", file=sys.stderr)
        return 2
    operation = sys.argv[1]
    action = " ".join(sys.argv[2:]).strip() if operation == "step" else None
    if operation == "step" and not action:
        print("step requires an ALFWorld action", file=sys.stderr)
        return 2
    return request(operation, action)


if __name__ == "__main__":
    raise SystemExit(main())
'''
