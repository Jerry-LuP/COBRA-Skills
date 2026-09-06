from __future__ import annotations

from cobras.task_envs.cobras_types import CobrasDriverContext
from cobras.task_envs.prompt_task_driver import build_prompt_task_command


def build_command(context: CobrasDriverContext) -> list[str]:
    return build_prompt_task_command(context)
