# COBRA-Skills

COBRA-Skills optimizes reusable task skills with a fixed-size evolutionary
population and an NN + LinearUCB bandit. The bandit allocates target-model
evaluations among current skills; scheduled evolutionary updates regenerate,
mutate, and cross over skills using a teacher model.

This release contains one model-agnostic experiment interface, six benchmark
adapters, fixed split manifests, and benchmark assets that can be redistributed.
Model names and experiment settings live in YAML. Endpoints and API keys live
in an env file.

## Overview

![COBRA-Skills framework](img/frameworks.png)

**Overview of COBRA-Skills.** At each round, skills in $\mathcal{P}_t$ are
prioritized using a neural reward predictor and LinearUCB bonus based on
$\mathcal{H}_{t-1}$ ①, and the highest-priority skill is evaluated by
the target agent ②. The resulting reward and trajectories update the
optimization history and scoring model ③. At scheduled population
updates (performed periodically rather than after every round), new skills are
evolved from accumulated evidence ④, and the updated scores identify
low-priority skills, which are replaced by the newly evolved skills to form
$\mathcal{P}_{t+1}$ ⑤. Dashed borders and arrows indicate scheduled-only
operations.

## Main Results

![COBRA-Skills main performance results](img/cobra-skills%20main%20results.png)

## Supported Tasks

| Dataset | Train | Test | Default reward |
| --- | ---: | ---: | --- |
| DocVQA | 50 | 100 | hard |
| LiveMath | 50 | 100 | hard |
| SearchQA | 50 | 100 | hard |
| SocialMaze-Hard | 50 | 100 | soft |
| SpreadsheetBench | 50 | 100 | hard |
| ALFWorld | 50 | 100 | hard |

ALFWorld uses at most 30 environment steps per episode in the default config.

## Install

Install the package and its Python dependencies in the environment you intend
to use:

```bash
cd cobra-skills
python -m pip install -e .
```

ALFWorld has an optional dependency group:

```bash
python -m pip install -e '.[alfworld]'
```

The release runner does not activate Conda environments, start model servers,
or install external harness CLIs. When using `codex` or `claude_code`, the
corresponding executable must already be available on `PATH`, or its path
must be set in `configs/default.yaml`.

## Prepare Data

SpreadsheetBench, SocialMaze-Hard, and ALFWorld data for the released splits
are included in this repository and require no separate materialization step.
They remain under their respective upstream licenses.

SearchQA, DocVQA, and LiveMath payloads are intentionally excluded because of
their redistribution terms. Fixed, ordered split manifests are included under
`data/manifests/`; each materializer writes only the 50-record training split
and 100-record test split used by COBRA-Skills.

Install the optional data dependencies:

```bash
python -m pip install -e '.[data]'
```

Materialize SearchQA and DocVQA from their Hugging Face mirrors:

```bash
cobra-materialize-data searchqa
cobra-materialize-data docvqa
```

LiveMath is not downloaded automatically because the monthly benchmark payload
does not currently have a clear redistribution license. Obtain the upstream
`qa_*_final.json` files, then apply the released split and deterministic
choice transformation:

```bash
cobra-materialize-data livemath --source-dir /path/to/livemath/monthly/files
```

The commands write to the paths expected by `configs/default.yaml`. Use
`--force` to replace an existing materialization. By downloading or supplying
benchmark files, you remain responsible for complying with each upstream
dataset's terms. See [data/README.md](data/README.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact scope, attribution,
source links, and modification notes.

## Configure Models

Create the private runtime env file:

```bash
cp .env.example .env
```

`.env` contains only endpoints and credentials:

```dotenv
TARGET_BASE_URL=
TARGET_API_KEY=

TEACHER_BASE_URL=
TEACHER_API_KEY=

EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
```

Set each base URL to the API root supplied by the corresponding provider or
self-hosted server. Teacher calls use an OpenAI-compatible chat-completions
endpoint, and embedding calls use an OpenAI-compatible embeddings endpoint.
The target protocol depends on the selected harness, as described below.

Set model names, providers, reasoning, token limits, workers, trials, and
algorithm parameters in [configs/default.yaml](configs/default.yaml).
Environment variables set by the calling shell take precedence over values in
the env file.

The default roles are:

- target: `qwen3.6-35b-a3b`, served locally, no reasoning, thinking disabled;
- teacher: `gpt-5.5`, medium reasoning;
- embedding: `qwen/qwen3-embedding-4b`.

To use an OpenAI-compatible local target, change the YAML target provider to
`local`, set its model name, and point `TARGET_BASE_URL` at the server.

The checked-in algorithm defaults run the full 30-round COBRAS configuration:
10 initial skills, a fixed pool of 10, pruning 3 arms on the logarithmic
schedule, sample size 8, `nu=0.1`, and `lambda=0.03`. Initial-pool evaluation is
disabled by default with `experiment.initial_eval: none`; this skips only the
separate initial-skill scoring stage, not the optimization rounds.

## Run

Run one dataset and one trial end to end:

```bash
cobras run searchqa --trials 1
```

Run all configured datasets and three trials:

```bash
cobras run --trials 1 2 3
```

The repository also includes a thin wrapper that only forwards arguments to
the Python CLI:

```bash
bash scripts/run_experiment.sh run searchqa --trials 1
```

The five subcommands can be run independently:

```bash
cobras baseline searchqa
cobras init searchqa --trials 1 2 3
cobras optimize searchqa --trials 1 2 3
cobras test searchqa --trials 1 2 3 --test-target both
cobras run searchqa --trials 1 2 3
```

Common command-line overrides include:

```bash
cobras run searchqa \
  --trials 1 \
  --workers 30 \
  --target-max-tokens 4096 \
  --teacher-max-tokens 16384 \
  --rounds 30 \
  --pool-size 10 \
  --prune-count 3 \
  --nu 0.1 \
  --lambda 0.03
```

Any YAML field can also be overridden with a dotted key:

```bash
cobras run socialmaze_hard \
  --set datasets.socialmaze_hard.max_completion_tokens=16384 \
  --set algorithm.generate_parallel=5
```

Use `--dry-run` to validate configuration and inspect all generated commands
without making model calls.

## Target Harnesses

Select the target execution backend with `harness.backend` in YAML or
`--harness`:

| Value | Behavior |
| --- | --- |
| `native` | Direct OpenAI-compatible chat completion calls |
| `codex` | One Codex CLI session per benchmark item |
| `claude_code` | One Claude Code CLI session per benchmark item |

Examples:

```bash
cobras run searchqa --trials 1 --harness native
cobras run spreadsheetbench --trials 1 --harness codex
cobras run docvqa --trials 1 --harness claude_code
```

For tool-using tasks, a self-hosted target used through `codex` or
`claude_code` must be served with tool calling enabled. The exact parser is
model- and server-version-specific. For example, the relevant vLLM options
have the following form:

```bash
vllm serve /path/to/model --served-model-name model_name \
  --enable-auto-tool-choice --tool-call-parser parser_name
```

Choose the parser supported by the target model. The configured target model
name must exactly match the served model ID, including case. The `native`
harness does not require tool calling unless the selected task itself uses
tools.

Before the first Codex run, build its minimal chroot from the active Python
environment and installed Codex CLI:

```bash
cobra-codex-setup
```

The setup and Codex runs require Linux and root access for `chroot`. Each item
runs under a fresh unprivileged UID with only its task workspace exposed.
Teacher and embedding secrets are not passed to the target subprocess. The
generated rootfs is stored under `.codex_chroot/` and is not committed. Rebuild
it with `--force` after changing the Python environment or Codex installation.
From a source checkout, the equivalent command is
`python scripts/harness_scripts/setup_codex_chroot.py`.

Set `harness.codex.executable: codex` only when the host already provides a
working sandbox for the plain Codex CLI. The release default uses
`scripts/harness_scripts/codex_chroot_exec.py` because container hosts commonly
deny Codex's nested bubblewrap mount namespace.

Claude Code uses a restricted tool allowlist and non-interactive `dontAsk`
permissions. For untrusted target models, place Claude Code inside an OS-level
container or sandbox as an additional boundary.

The target endpoint used by an exec harness must support that CLI's wire
protocol. The native harness expects OpenAI-compatible chat completions.
`TARGET_BASE_URL` may use the same trailing `/v1` form for every harness;
COBRA-Skills removes that suffix before passing the endpoint to Claude Code,
which appends its own `/v1/messages` route.

## Output Layout

The default output root is `outputs/`:

```text
outputs/
  baselines/<dataset>/target_<model>/harness_<backend>/train_<n>/
  init_skills/<dataset>/target_<target>__teacher_<teacher>/
    harness_<backend>/trial_001/
  runs/<dataset>/target_<target>__teacher_<teacher>/
    harness_<backend>/<experiment>/trial_001/
      request.json
      resolved_config.json
      history.jsonl
      rounds/
      optimized_skill/skill.md
      test_eval/
        baseline/
        optimized/
      token_usage/
```

Baseline, initial-skill, optimization, and test stages each write token events
and summaries under their own artifact root. Token summaries separate student
and teacher input/output tokens.

Resume is enabled by default. Completed stages are skipped. Every artifact root
contains a request manifest; reusing the same path with incompatible model,
dataset, or algorithm parameters fails instead of mixing experiments. Endpoint
changes do not alter experiment identity, so equivalent local ports can resume
the same run.

## Initial Skills

Initial skills are generated from the target model's no-skill training
trajectories. They are not bundled from historical experiments.

Generated initial skills are written to `outputs/init_skills/`; the empty
`datasets/<task>/init_skills/` directories exist only as low-level defaults.

`experiment.initial_eval` controls initial-pool evaluation:

- `all`: evaluate every trial's initial pool;
- `first_trial`: evaluate only trial 1;
- `none`: skip initial-pool evaluation, the default.

Skipping initial evaluation does not seed the online NN/UCB history with
synthetic rewards. Online history begins with skills selected by the main
optimization loop.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q tests
```

## License

Unless a file or directory says otherwise, COBRA-Skills original source code
is licensed under the [Apache License 2.0](LICENSE).

Third-party code, benchmark data, and materialized downloads are not
relicensed under Apache-2.0. Their original terms continue to apply. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[data/README.md](data/README.md).
