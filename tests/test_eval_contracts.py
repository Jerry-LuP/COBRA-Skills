from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from cobras.cobras_core.config import DATASETS, get_dataset
from cobras.cobras_core.llm import ProviderConfig
from cobras.task_envs.eval_types import EvalConfig
from cobras.task_envs.eval_runtime import configure_chat_target
from cobras.task_envs.registry import load_evaluator
from cobras.task_envs.cobras_types import CobrasDriverContext
from cobras.scripts.run_cobras import ensure_initial_eval, shared_baseline_root
from cobras.scripts.generate_skill_from_results import pool_sample_seed, pool_skill_dir


def _provider() -> ProviderConfig:
    return ProviderConfig(
        provider="local",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env="LOCAL_LLM_API_KEY",
        model="test-model",
    )


def _config(dataset: str, out_root: Path) -> EvalConfig:
    return EvalConfig(
        dataset=get_dataset(dataset),
        provider=_provider(),
        split="train",
        limit=1,
        workers=1,
        out_root=out_root,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvalContractTests(unittest.TestCase):
    def test_shared_baseline_root_is_dataset_model_and_provider_specific(self) -> None:
        path = shared_baseline_root("livemath", "openai/gpt-5.4-nano", "openrouter")
        self.assertEqual(
            path.relative_to(path.parents[4]),
            Path(
                "results/baselines/livemath/target_gpt-5.4-nano__provider_openrouter/"
                "train50_direct"
            ),
        )

    def test_skill_generator_pool_uses_distinct_dirs_and_sample_seeds(self) -> None:
        root = Path("/tmp/init_skills")
        self.assertEqual(pool_skill_dir(root, 1), root / "seed_01")
        self.assertEqual(pool_skill_dir(root, 10), root / "seed_10")
        self.assertEqual(pool_sample_seed(42, 1), 42)
        self.assertEqual(pool_sample_seed(42, 2), 1051)

    def test_initial_eval_can_be_skipped_without_assigning_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = root / "pool"
            for name in ("seed_01", "seed_02"):
                skill_dir = pool / name
                skill_dir.mkdir(parents=True)
                (skill_dir / "skill.md").write_text("# Skill\n", encoding="utf-8")
            args = Namespace(
                dataset="alfworld",
                initial_eval_summary=None,
                initial_pool_root=pool,
                pool_size=2,
                skip_initial_eval=True,
            )
            summary_path = ensure_initial_eval(args, {}, root / "trial")
            payload = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertTrue(payload["evaluation_skipped"])
            self.assertEqual(len(payload["results"]), 2)
            self.assertNotIn("avg_hard_reward", payload["results"][0])
            self.assertNotIn("avg_soft_reward", payload["results"][0])

    def test_openai_compatible_provider_uses_openai_backend(self) -> None:
        provider = ProviderConfig(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            model="openai/gpt-5.4-nano",
        )
        config = EvalConfig(
            dataset=get_dataset("searchqa"),
            provider=provider,
            split="train",
            limit=1,
            workers=1,
            out_root=Path("/tmp/unused"),
            api_key="test-key",
        )
        with patch("cobras.task_envs.eval_runtime.set_target_backend") as set_backend, patch(
            "cobras.task_envs.eval_runtime.configure_openai"
        ) as configure_openai:
            backend, _, _, _ = configure_chat_target(
                config,
                default_max_completion_tokens=2048,
                default_timeout=300,
            )

        self.assertEqual(backend, "openai_chat")
        set_backend.assert_called_once_with("openai_chat")
        configure_openai.assert_called_once()

    def test_local_provider_uses_openai_compatible_backend(self) -> None:
        provider = ProviderConfig(
            provider="local",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="TARGET_API_KEY",
            model="local-model",
        )
        config = EvalConfig(
            dataset=get_dataset("searchqa"),
            provider=provider,
            split="train",
            limit=1,
            workers=1,
            out_root=Path("/tmp/unused"),
            api_key="",
        )
        with patch("cobras.task_envs.eval_runtime.set_target_backend") as set_backend, patch(
            "cobras.task_envs.eval_runtime.configure_openai"
        ) as configure_openai:
            backend, _, _, _ = configure_chat_target(
                config,
                default_max_completion_tokens=2048,
                default_timeout=300,
            )

        self.assertEqual(backend, "openai_chat")
        set_backend.assert_called_once_with("openai_chat")
        configure_openai.assert_called_once()

    def test_all_dataset_modules_are_registered(self) -> None:
        for name, dataset in DATASETS.items():
            self.assertTrue(callable(load_evaluator(name)))
            self.assertTrue(dataset.evaluator_module.startswith("cobras.task_envs."))
            self.assertTrue(dataset.skill_generator_module.startswith("cobras.task_envs."))
            self.assertTrue(dataset.skill_mutator_module.startswith("cobras.task_envs."))
            self.assertTrue(dataset.cobras_driver_module.startswith("cobras.task_envs."))

    def test_all_cobras_drivers_build_standalone_commands(self) -> None:
        args = Namespace(
            teacher_model="openai/gpt-5.4",
            api_key="",
            rounds=30,
            pool_size=10,
            prune_count=3,
            prune_interval=3,
            prune_schedule="log",
            prune_log_threshold=0.35,
            prune_cooldown_rounds=None,
            prune_max_interval=None,
            regen_sample_size=8,
            rollout_sample_size=8,
            generate_parallel=3,
            eval_workers=15,
            max_turns=37,
            eval_limit=50,
            nu=0.1,
            lambda_=0.03,
            embedding_backend="openrouter",
            embedding_model="qwen/qwen3-embedding-4b",
            embedding_dim=2560,
            reward_field="",
            resume=True,
            test_mode=False,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in DATASETS.values():
                context = CobrasDriverContext(
                    args=args,
                    dataset=dataset,
                    provider=_provider(),
                    teacher_provider=_provider(),
                    common_args=("--rounds", "30"),
                    out_root=root / dataset.name,
                    baseline_root=root / "baseline",
                    initial_eval_summary=root / "init_eval.json",
                    initial_pool_root=dataset.init_skills_root,
                )
                command = import_module(dataset.cobras_driver_module).build_command(context)
                rendered = " ".join(command)
                self.assertIn("cobras.", rendered)
                self.assertNotIn(".sh", rendered)
                self.assertNotIn(" -m scripts.", rendered)
                self.assertIn(f"--reward-field {dataset.reward_field}", rendered)
                if dataset.name in {"alfworld", "docvqa", "livemath", "searchqa", "spreadsheetbench"}:
                    self.assertIn("cobras.scripts.cobras_dynamic", rendered)
                    self.assertIn(f"--dataset {dataset.name}", rendered)
                    from cobras.scripts.cobras_dynamic import parse_args as parse_dynamic_args

                    with patch("sys.argv", ["cobras_dynamic", *command[3:]]):
                        parsed = parse_dynamic_args()
                    self.assertEqual(parsed.dataset, dataset.name)
                    self.assertEqual(parsed.initial_pool_root, dataset.init_skills_root)
                    self.assertEqual(parsed.reward_field, dataset.reward_field)
                    self.assertEqual(parsed.max_turns, 37)
                    if dataset.name == "livemath":
                        self.assertEqual(parsed.model, context.provider.model)
                        self.assertEqual(parsed.teacher_model, context.teacher_provider.model)

    def test_docvqa_eval_artifacts(self) -> None:
        result = {
            "id": "1",
            "question": "q",
            "gold_answer": ["a"],
            "predicted_answer": "a",
            "hard": 1,
            "soft": 1.0,
            "agent_ok": True,
            "fail_reason": "",
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "docvqa"

            def fake_run(*_args, **_kwargs):
                _write_jsonl(out_root / "results.jsonl", [result])
                return [result]

            with patch(
                "cobras.task_envs.docvqa.eval.configure_chat_target",
                return_value=("openai_chat", None, 4096, 600),
            ), patch(
                "cobras.task_envs.docvqa.eval.DocVQADataLoader.load_split_items",
                return_value=[{"id": "1", "question": "q", "answers": ["a"]}],
            ), patch("cobras.task_envs.docvqa.eval.run_batch", side_effect=fake_run):
                artifacts = load_evaluator("docvqa")(_config("docvqa", out_root))
            self.assertEqual(artifacts.summary["hard_acc"], 1.0)
            self.assertTrue(artifacts.brief_result_path.exists())

    def test_searchqa_eval_artifacts(self) -> None:
        result = {
            "id": "1",
            "question": "q",
            "gold_answers": ["a"],
            "predicted_answer": "a",
            "hard": 1,
            "soft": 1.0,
            "agent_ok": True,
            "fail_reason": "",
            "n_turns": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "searchqa"

            def fake_run(*_args, **_kwargs):
                _write_jsonl(out_root / "results.jsonl", [result])
                return [result]

            with patch(
                "cobras.task_envs.searchqa.eval.configure_chat_target",
                return_value=("openai_chat", "low", 2048, 600),
            ), patch(
                "cobras.task_envs.searchqa.eval.SearchQADataLoader.load_split_items",
                return_value=[{"id": "1", "question": "q", "answers": ["a"]}],
            ), patch("cobras.task_envs.searchqa.eval.run_batch", side_effect=fake_run):
                artifacts = load_evaluator("searchqa")(_config("searchqa", out_root))
            self.assertEqual(artifacts.summary["hard_acc"], 1.0)
            self.assertTrue(artifacts.brief_result_path.exists())

    def test_searchqa_resume_retries_failed_agent_rows(self) -> None:
        failed = {"id": "1", "agent_ok": False, "hard": 0, "fail_reason": "connection error"}
        recovered = {"id": "1", "agent_ok": True, "hard": 1, "soft": 1.0}
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp)
            _write_jsonl(out_root / "results.jsonl", [failed])
            with patch("cobras.task_envs.searchqa.rollout.process_one", return_value=recovered) as process_one:
                from cobras.task_envs.searchqa.rollout import run_batch

                results = run_batch(
                    items=[{"id": "1", "question": "q", "answers": ["a"]}],
                    out_root=str(out_root),
                    skill_content="",
                    workers=1,
                )

            self.assertEqual(results, [recovered])
            process_one.assert_called_once()
            stored = [json.loads(line) for line in (out_root / "results.jsonl").read_text().splitlines()]
            self.assertEqual(stored, [recovered])

    def test_spreadsheetbench_eval_artifacts(self) -> None:
        result = {
            "id": "1",
            "hard": 1,
            "soft": 1.0,
            "llm_ok": True,
            "code_ok": True,
            "exec_ok": True,
            "n_turns": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "spreadsheetbench"
            with patch(
                "cobras.task_envs.spreadsheetbench.eval.configure_chat_target",
                return_value=("openai_chat", "low", 8192, 600),
            ), patch(
                "cobras.task_envs.spreadsheetbench.eval._load_items",
                return_value=([{"id": "1", "instruction": "q"}], Path(tmp)),
            ), patch(
                "cobras.task_envs.spreadsheetbench.eval.run_spreadsheet_batch_codegen",
                return_value=[result],
            ):
                artifacts = load_evaluator("spreadsheetbench")(_config("spreadsheetbench", out_root))
            self.assertEqual(artifacts.summary["hard_acc"], 1.0)
            self.assertTrue(artifacts.results_path.exists())

    def test_socialmaze_eval_artifacts(self) -> None:
        result = {
            "id": "1",
            "task": "hidden_role",
            "hard": 1,
            "soft": 1.0,
            "agent_ok": True,
            "gold_answer": "gold",
            "gold_parsed": {"criminal_player": 2, "role": "Investigator"},
            "predicted_answer": "pred",
            "predicted_parsed": {"criminal_player": 2, "role": "Investigator"},
            "fail_reason": "",
            "duration": 0.1,
            "usage": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "socialmaze"
            with patch(
                "cobras.task_envs.socialmaze_hard.eval.configure_chat_target",
                return_value=("openai_chat", "low", 2048, 240),
            ), patch(
                "cobras.task_envs.socialmaze_hard.eval.load_items",
                return_value=[{"id": "1"}],
            ), patch(
                "cobras.task_envs.socialmaze_hard.eval.run_one",
                return_value=result,
            ):
                artifacts = load_evaluator("socialmaze_hard")(_config("socialmaze_hard", out_root))
            self.assertEqual(artifacts.summary["hard_acc"], 1.0)
            self.assertEqual(artifacts.summary["avg_soft"], 1.0)

    def test_livemath_eval_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_root = Path(tmp) / "livemath"

            def fake_run(args):
                root = args.out_root / args.run_name
                root.mkdir(parents=True)
                (root / "summary.json").write_text('{"avg_hard_reward": 1.0}', encoding="utf-8")
                _write_jsonl(root / "trial_results.jsonl", [{"hard_reward": 1.0}])
                return root

            with patch("cobras.task_envs.livemath.eval.run", side_effect=fake_run):
                artifacts = load_evaluator("livemath")(_config("livemath", out_root))
            self.assertEqual(artifacts.summary["avg_hard_reward"], 1.0)
            self.assertTrue(artifacts.results_path.exists())


if __name__ == "__main__":
    unittest.main()
