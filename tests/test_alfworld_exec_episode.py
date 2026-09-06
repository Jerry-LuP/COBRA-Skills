from __future__ import annotations

import json
import socket

import pytest

from cobras.task_envs.alfworld.exec_episode import AlfworldExecEpisode, serve_episode


class _FakeEnvManager:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def step(self, responses: list[str]):
        response = responses[0]
        action = response.split("<action>", 1)[1].split("</action>", 1)[0]
        self.actions.append(action)
        won = action == "finish task"
        observation = {
            "text": [f"observation after {action}"],
            "anchor": [f"raw after {action}"],
            "image": [None],
        }
        return observation, [1.0 if won else 0.0], [won], [
            {
                "won": won,
                "is_action_valid": True,
                "admissible_commands": ["look around", "finish task"],
            }
        ]


def _controller(*, max_steps: int = 3) -> AlfworldExecEpisode:
    return AlfworldExecEpisode(
        env_manager=_FakeEnvManager(),
        observation={"text": ["initial"], "anchor": ["raw initial"], "image": [None]},
        info={"admissible_commands": ["look around", "finish task"]},
        max_steps=max_steps,
    )


def _request(port: int, payload: dict) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        response = sock.makefile("rb").readline()
    return json.loads(response)


def test_episode_controller_steps_real_environment_until_win() -> None:
    controller = _controller()

    observed = controller.observe()
    assert observed["observation"] == "raw initial"
    assert observed["admissible_actions"] == ["look around", "finish task"]
    first = controller.step("look around")
    assert first["step"] == 1
    assert first["done"] is False
    assert first["observation"] == "raw after look around"

    final = controller.step("finish task")
    assert final["step"] == 2
    assert final["done"] is True
    assert final["won"] is True
    assert controller.env_manager.actions == ["look around", "finish task"]
    assert [row["action"] for row in controller.steps] == ["look around", "finish task"]


def test_episode_controller_enforces_step_limit_and_action_contract() -> None:
    controller = _controller(max_steps=1)

    with pytest.raises(ValueError, match="forbidden"):
        controller.step("<action>look</action>")
    state = controller.step("look around")
    assert state["done"] is True
    assert state["step_limit_reached"] is True
    assert state["environment_done"] is False


def test_episode_bridge_requires_random_token() -> None:
    controller = _controller()
    with serve_episode(controller) as (port, token):
        denied = _request(port, {"token": "wrong", "operation": "observe"})
        observed = _request(port, {"token": token, "operation": "observe"})
        stepped = _request(
            port,
            {"token": token, "operation": "step", "action": "finish task"},
        )

    assert denied["ok"] is False
    assert "invalid episode token" in denied["error"]
    assert observed["ok"] is True
    assert stepped["won"] is True
