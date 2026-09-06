"""Standalone ALFWorld text-environment rollout used by COBRAS."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import time

from cobras.cobras_core.model import (
    chat_target,
    get_target_backend,
    get_target_deployment,
    is_target_exec_backend,
)
from cobras.cobras_core.model.exec_harness import prepare_workspace, render_skill_md, run_target_exec
from cobras.task_envs.alfworld.exec_episode import (
    AlfworldExecEpisode,
    build_client_script,
    serve_episode,
)
from cobras.task_envs.eval_errors import is_timeout_error, raise_evaluation_error

# Constants

TASKS = [
    "pick_and_place_simple",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]

# Helpers


def _get_task_type(gamefile: str) -> str:
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def _extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _extract_think(model_response: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def _build_skill_prompt(skill_content: str) -> str:
    """Build the skill section to inject into the agent's system prompt."""
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


def _append_diagnostic_instruction(prompt: str, diagnostic_instruction: str) -> str:
    if not diagnostic_instruction or not diagnostic_instruction.strip():
        return prompt
    return f"{prompt}\n\n## Training Readout\n{diagnostic_instruction.strip()}\n"


def _run_exec_action(
    *,
    out_root: str,
    task_id: str,
    step_idx: int,
    prompt: str,
    skill_content: str,
    timeout: int,
) -> tuple[str, str]:
    if not out_root:
        raise ValueError("ALFWorld exec backends require out_root for isolated workspaces.")
    work_dir = os.path.join(
        out_root,
        "predictions",
        _safe_name(task_id),
        "exec_backend",
        f"step_{step_idx + 1:03d}",
    )
    skill_md = render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for choosing the next ALFWorld action.",
        preamble=(
            "Use this skill as reusable ALFWorld strategy. Base the next action only "
            "on the current task state in task.md."
        ),
    )
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=(
            "## Agent Role\n"
            "You are an expert agent operating in the ALFRED embodied environment.\n\n"
            "## Current State and Action Contract\n"
            f"{prompt}\n\n"
            "Return exactly one next environment action inside <action>...</action>. "
            "You may put concise private reasoning inside <think>...</think>."
        ),
    )
    return run_target_exec(
        work_dir=work_dir,
        prompt=(
            "Read task.md and .agents/skills/cobras-target/SKILL.md, then choose "
            "the single best next ALFWorld action. Return <think>...</think>"
            "<action>...</action>."
        ),
        model=get_target_deployment(),
        timeout=timeout,
        stage="alfworld_rollout",
    )


def _exec_session_mode() -> str:
    mode = os.environ.get(
        "ALFWORLD_EXEC_SESSION_MODE",
        os.environ.get("ALFWORLD_CODEX_SESSION_MODE", "action"),
    ).strip().lower()
    if mode not in {"action", "episode"}:
        raise ValueError(
            "ALFWORLD_EXEC_SESSION_MODE must be 'action' or 'episode', "
            f"got {mode!r}"
        )
    return mode


def _codex_session_mode() -> str:
    """Backward-compatible alias for older Codex launchers and tests."""
    return _exec_session_mode()


def _run_exec_episode_batch(
    *,
    env_manager,
    obs: dict,
    infos: list[dict],
    env_meta: list[dict],
    conversations: list[list[dict]],
    skill_content: str,
    max_steps: int,
    out_root: str,
    request_timeout: int | None,
    task_timeout: int | None,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
    result_ids: list[str] | None,
) -> list[dict]:
    """Run one complete ALFWorld episode inside one exec-harness session."""
    if len(env_meta) != 1:
        raise ValueError(
            "ALFWorld exec episode mode requires one environment per evaluator worker."
        )
    if not out_root:
        raise ValueError("ALFWorld exec episode mode requires out_root.")

    task_id = str(result_ids[0]) if result_ids else "env_000"
    conv_dir = os.path.join(out_root, "predictions", _safe_name(task_id))
    work_dir = os.path.join(conv_dir, "exec_backend")
    meta = env_meta[0]
    controller = AlfworldExecEpisode(
        env_manager=env_manager,
        observation=obs,
        info=infos[0] if infos else {},
        max_steps=max_steps,
    )
    skill_md = render_skill_md(
        skill_content,
        description="Reusable strategy for completing a real ALFWorld episode.",
        preamble=(
            "Use this skill while interacting with the environment through "
            "alfworld_tool.py. Ground every action in the latest observation."
        ),
    )
    diagnostic = ""
    if diagnostic_mode and diagnostic_instruction.strip():
        diagnostic = f"\n\n## Training Readout\n{diagnostic_instruction.strip()}\n"
    task_text = (
        "## Agent Role\n"
        "You are controlling one real ALFWorld text-environment episode.\n\n"
        "## Task\n"
        f"Type: {meta['task_type']}\n"
        f"Description: {meta['task_description']}\n\n"
        "## Environment Tool Contract\n"
        "The environment is not simulated in this document. Use only these commands:\n"
        "- `python alfworld_tool.py observe`\n"
        "- `python alfworld_tool.py step \"EXACT ADMISSIBLE ACTION\"`\n"
        "Each command returns JSON containing the latest observation, reward, and done flag. "
        f"The bridge enforces a maximum of {max_steps} environment steps.\n"
        f"{diagnostic}"
    )
    prompt = (
        "Read task.md and `.agents/skills/cobras-target/SKILL.md` directly. "
        "Run `python alfworld_tool.py observe`, then repeatedly choose one action "
        "from the latest admissible-action list and run "
        "`python alfworld_tool.py step \"ACTION\"`. Continue until the returned JSON "
        "has `done: true`; do not stop early. After completion, briefly report whether "
        "the environment says `won: true`."
    )

    response = ""
    model_error = ""
    task_timed_out = False
    episode_timeout = (
        int(task_timeout)
        if task_timeout and task_timeout > 0
        else int(request_timeout or 600)
    )
    try:
        with serve_episode(controller) as (port, token):
            prepare_workspace(
                work_dir=work_dir,
                skill_md=skill_md,
                task_text=task_text,
                extra_files={
                    "alfworld_tool.py": build_client_script(port=port, token=token),
                },
            )
            response, _ = run_target_exec(
                work_dir=work_dir,
                prompt=prompt,
                model=get_target_deployment(),
                timeout=episode_timeout,
                stage="alfworld_rollout_episode",
            )
    except Exception as exc:  # noqa: BLE001
        if not is_timeout_error(exc):
            raise_evaluation_error("ALFWorld", task_id, exc)
        task_timed_out = True
        model_error = f"{type(exc).__name__}: {exc}"

    conversations[0].extend(controller.steps)
    if not controller.done and not model_error:
        model_error = "Exec-harness session exited before the ALFWorld episode completed"

    won = controller.won
    fail_reason = ""
    if not won:
        if task_timed_out:
            fail_reason = f"Task timeout after {episode_timeout}s"
        elif controller.limit_reached:
            fail_reason = f"Timeout after {max_steps} steps"
        elif controller.environment_done:
            fail_reason = "Episode ended without completing the task"
        else:
            fail_reason = model_error or "Exec-harness session ended before episode completion"
    if model_error:
        fail_reason = f"{fail_reason}; {model_error}" if fail_reason else model_error

    os.makedirs(conv_dir, exist_ok=True)
    conversation_path = os.path.join(conv_dir, "conversation.json")
    with open(conversation_path, "w", encoding="utf-8") as f:
        json.dump(conversations[0], f, ensure_ascii=False, indent=2)

    return [
        {
            "id": task_id,
            "hard": 1 if won else 0,
            "soft": 1.0 if won else 0.0,
            "n_turns": len(controller.steps),
            "fail_reason": fail_reason,
            "agent_ok": not model_error and not task_timed_out,
            "task_timed_out": task_timed_out,
            "task_type": meta["task_type"],
            "gamefile": meta["gamefile"],
            "task_description": meta["task_description"],
            "instruction_type": meta["task_type"],
            "codex_session_mode": "episode",
            "codex_final_response": response,
            "artifact_paths": {
                "conversation_json": os.path.relpath(conversation_path, out_root),
            },
        }
    ]


def _resolve_alfworld_gamefile(gamefile: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(gamefile)))
    if os.path.isabs(path):
        return path

    data_root = os.environ.get("ALFWORLD_DATA", "").strip()
    if not data_root:
        return path

    root = os.path.expanduser(os.path.expandvars(data_root))
    return os.path.abspath(os.path.join(root, path))


def _resolve_alfworld_gamefiles(gamefiles: list[str] | None) -> list[str] | None:
    if gamefiles is None:
        return None
    return [_resolve_alfworld_gamefile(gamefile) for gamefile in gamefiles]


# Environment builder


def build_alfworld_env(
    env_num: int,
    eval_dataset: str = "eval_out_of_distribution",
    seed: int = 42,
    is_train: bool = False,
    specific_gamefiles: list[str] | None = None,
):
    """Build ALFWorld environment manager.

    Args:
        env_num: number of parallel environments
        eval_dataset: 'eval_in_distribution' or 'eval_out_of_distribution' or train
        seed: random seed
        is_train: whether to use training set

    Returns:
        env_manager: AlfWorldEnvironmentManager instance
    """
    from functools import partial

    from omegaconf import OmegaConf

    from cobras.task_envs.alfworld.vendor.alfworld_envs import build_alfworld_envs
    from cobras.task_envs.alfworld.vendor.alfworld_projection import alfworld_projection
    from cobras.task_envs.alfworld.vendor.env_manager import AlfWorldEnvironmentManager

    HERE = os.path.dirname(os.path.abspath(__file__))

    alf_config_path = os.path.join(HERE, "vendor", "config_tw.yaml")
    env_kwargs = {"eval_dataset": eval_dataset}
    resolved_gamefiles = _resolve_alfworld_gamefiles(specific_gamefiles)

    envs = build_alfworld_envs(
        alf_config_path,
        seed=seed,
        env_num=env_num,
        group_n=1,
        is_train=is_train,
        env_kwargs=env_kwargs,
        resources_per_worker=None,
        gamefiles=resolved_gamefiles,
    )

    config = OmegaConf.create(
        {
            "env": {
                "history_length": 2,
                "env_name": "alfworld/AlfredTWEnv",
            }
        }
    )

    projection_f = partial(alfworld_projection)
    env_manager = AlfWorldEnvironmentManager(envs, projection_f, config)
    return env_manager


# Batch rollout


def run_alfworld_batch(
    env_manager,
    skill_content: str,
    max_steps: int,
    out_root: str = "",
    max_api_workers: int = 8,
    max_completion_tokens: int = 16384,
    request_timeout: int | None = None,
    task_timeout: int | None = None,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    result_ids: list[str] | None = None,
) -> list[dict]:
    """Run a batch of ALFWorld episodes.

    Returns a list of result dictionaries consumed by the COBRAS evaluator:
    [
        {
            "id": "<env_idx>_<gamefile_hash>",
            "hard": 0 or 1,
            "soft": 0.0 or 1.0,
            "n_turns": <int>,
            "fail_reason": "<str>",
            "agent_ok": True,
            "task_type": "<str>",
            "gamefile": "<str>",
            "task_description": "<str>",
        },
        ...
    ]

    Also saves conversation.json per environment in out_root/predictions/<task_id>/
    """
    skill_prompt = _build_skill_prompt(skill_content)

    obs, infos = env_manager.reset({})
    env_num = len(obs["text"])
    env_dones = [False] * env_num
    overall_success = [False] * env_num
    model_error_counts = [0] * env_num
    task_timed_out = [False] * env_num
    task_started_at = [time.monotonic()] * env_num

    # Build per-env metadata
    env_meta: list[dict] = []
    for i in range(env_num):
        gamefile = infos[i].get("extra.gamefile", "") if isinstance(infos[i], dict) else ""
        task_type = _get_task_type(gamefile)
        # Extract task description from initial observation
        task_desc = ""
        anchor_text = obs["anchor"][i] if "anchor" in obs else ""
        task_start = anchor_text.find("Your task is to: ")
        if task_start != -1:
            task_desc = anchor_text[task_start + len("Your task is to: "):].strip()

        env_meta.append({
            "gamefile": gamefile,
            "task_type": task_type,
            "task_description": task_desc,
        })

    conversations: list[list[dict]] = [
        [
            {
                "type": "task",
                "task_description": env_meta[i]["task_description"],
                "task_type": env_meta[i]["task_type"],
                "initial_observation": obs["anchor"][i] if "anchor" in obs else "",
            }
        ]
        for i in range(env_num)
    ]

    if is_target_exec_backend() and _exec_session_mode() == "episode":
        return _run_exec_episode_batch(
            env_manager=env_manager,
            obs=obs,
            infos=infos,
            env_meta=env_meta,
            conversations=conversations,
            skill_content=skill_content,
            max_steps=max_steps,
            out_root=out_root,
            request_timeout=request_timeout,
            task_timeout=task_timeout,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            result_ids=result_ids,
        )


    for step_idx in range(max_steps):
        if all(env_dones):
            break

        if task_timeout and task_timeout > 0:
            now = time.monotonic()
            for i in range(env_num):
                if not env_dones[i] and now - task_started_at[i] >= task_timeout:
                    task_timed_out[i] = True
                    env_dones[i] = True
            if all(env_dones):
                break

        active_indices = [i for i in range(env_num) if not env_dones[i]]

        # Build prompts with skill injection
        prompts: dict[int, str] = {}
        for i in active_indices:
            prompt = obs["text"][i]
            if skill_prompt and not is_target_exec_backend():
                # Chat targets receive the skill inline; exec targets read SKILL.md.
                prompt = skill_prompt + "\n" + prompt
            if diagnostic_mode and diagnostic_instruction.strip():
                prompt = _append_diagnostic_instruction(prompt, diagnostic_instruction)
            prompts[i] = prompt

        # Call API in parallel
        actions = ["None"] * env_num

        def call_api(idx):
            try:
                effective_request_timeout = request_timeout
                if task_timeout and task_timeout > 0:
                    remaining = task_timeout - (time.monotonic() - task_started_at[idx])
                    if remaining <= 0:
                        return (
                            idx,
                            "<think>task timeout</think><action>look</action>",
                            f"task timeout after {task_timeout}s",
                            True,
                        )
                    # chat_target may make up to five attempts. Keep one slow
                    # request from consuming the entire remaining episode budget.
                    per_attempt_budget = max(1, int(remaining / 5))
                    effective_request_timeout = min(
                        effective_request_timeout or per_attempt_budget,
                        per_attempt_budget,
                    )
                if is_target_exec_backend():
                    task_id = (
                        str(result_ids[idx])
                        if result_ids and idx < len(result_ids)
                        else f"env_{idx:03d}"
                    )
                    response, _ = _run_exec_action(
                        out_root=out_root,
                        task_id=task_id,
                        step_idx=step_idx,
                        prompt=prompts[idx],
                        skill_content=skill_content,
                        timeout=effective_request_timeout or 300,
                    )
                else:
                    response, _ = chat_target(
                        system="You are an expert agent operating in the ALFRED Embodied Environment.",
                        user=prompts[idx],
                        max_completion_tokens=max_completion_tokens,
                        retries=5,
                        stage="rollout",
                        reasoning_effort=None,
                        timeout=effective_request_timeout,
                    )
                if task_timeout and time.monotonic() - task_started_at[idx] >= task_timeout:
                    return (
                        idx,
                        "<think>task timeout</think><action>look</action>",
                        f"task timeout after {task_timeout}s",
                        True,
                    )
                response = (response or "").strip()
                if not response:
                    return idx, "<think>empty model response</think><action>look</action>", "empty model response", False
                if _extract_action(response) is None:
                    return idx, "<think>missing action tag</think><action>look</action>", "missing action tag", False
                return idx, response, "", False
            except Exception as exc:  # noqa: BLE001
                if is_timeout_error(exc):
                    return (
                        idx,
                        "<think>task timeout</think><action>look</action>",
                        f"{type(exc).__name__}: {exc}",
                        True,
                    )
                raise

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_api_workers)
        try:
            futures = {executor.submit(call_api, i): i for i in active_indices}
            pending_futs = set(futures)
            while pending_futs:
                done, _ = concurrent.futures.wait(
                    pending_futs,
                    timeout=5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    pending_futs.remove(future)
                    try:
                        idx, response, model_error, timed_out = future.result()
                    except Exception as exc:  # noqa: BLE001
                        idx = futures[future]
                        if is_timeout_error(exc):
                            response = "<think>task timeout</think><action>look</action>"
                            model_error = f"{type(exc).__name__}: {exc}"
                            timed_out = True
                        else:
                            for pending_future in futures:
                                pending_future.cancel()
                            task_id = (
                                str(result_ids[idx])
                                if result_ids and idx < len(result_ids)
                                else f"env_{idx:03d}"
                            )
                            raise_evaluation_error("ALFWorld", task_id, exc)
                    actions[idx] = response
                    if model_error:
                        model_error_counts[idx] += 1
                    if timed_out:
                        task_timed_out[idx] = True
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # Save model responses before stepping
        model_responses = {i: actions[i] for i in active_indices}

        # Step environment
        obs, rewards, dones, infos = env_manager.step(actions)

        # Record trajectory
        for i in active_indices:
            step_record = {
                "step": step_idx,
                "action": _extract_action(model_responses[i]),
                "reasoning": _extract_think(model_responses[i]),
                "model_response": model_responses[i],
                "env_feedback": obs["anchor"][i] if "anchor" in obs else "",
                "reward": float(rewards[i]),
                "done": bool(dones[i]) or task_timed_out[i],
            }
            conversations[i].append(step_record)

        # Update done status
        for i in range(env_num):
            if env_dones[i]:
                continue
            if task_timed_out[i]:
                env_dones[i] = True
                continue
            if dones[i]:
                env_dones[i] = True
                won = bool(infos[i].get("won", False))
                overall_success[i] = won

    # Build results and save conversations
    results: list[dict] = []
    pred_dir = os.path.join(out_root, "predictions") if out_root else ""

    for i in range(env_num):
        gamefile = env_meta[i]["gamefile"]
        task_type = env_meta[i]["task_type"]
        task_desc = env_meta[i]["task_description"]
        n_turns = max(0, len(conversations[i]) - 1)
        won = overall_success[i]

        # Generate stable task ID from env index and gamefile
        task_id = str(result_ids[i]) if result_ids and i < len(result_ids) else f"env_{i:03d}"

        fail_reason = ""
        if not won:
            if task_timed_out[i]:
                fail_reason = f"Task timeout after {task_timeout}s"
            elif not env_dones[i]:
                fail_reason = f"Timeout after {max_steps} steps"
            else:
                fail_reason = "Episode ended without completing the task"
        if model_error_counts[i]:
            suffix = f"model response errors={model_error_counts[i]}"
            fail_reason = f"{fail_reason}; {suffix}" if fail_reason else suffix

        result = {
            "id": task_id,
            "hard": 1 if won else 0,
            "soft": 1.0 if won else 0.0,
            "n_turns": n_turns,
            "fail_reason": fail_reason,
            "agent_ok": model_error_counts[i] == 0 and not task_timed_out[i],
            "task_timed_out": task_timed_out[i],
            "task_type": task_type,
            "gamefile": gamefile,
            "task_description": task_desc,
            "instruction_type": task_type,
        }
        results.append(result)

        # Save conversation
        if pred_dir:
            conv_dir = os.path.join(pred_dir, _safe_name(task_id))
            os.makedirs(conv_dir, exist_ok=True)
            with open(os.path.join(conv_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(conversations[i], f, ensure_ascii=False, indent=2)
            result["artifact_paths"] = {
                "conversation_json": os.path.relpath(
                    os.path.join(conv_dir, "conversation.json"), out_root
                )
            }

    return results
