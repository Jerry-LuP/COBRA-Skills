# Third-Party Notices

The root Apache License 2.0 applies to original COBRA-Skills source code unless
a file or directory states otherwise. It does not relicense third-party code,
benchmark data, or data downloaded by the materialization command.

## SpreadsheetBench

- Scope: benchmark assets under `data/spreadsheetbench/` and
  `data/spreadsheetbench_id_split/`; derived portions of
  `task_envs/spreadsheetbench/evaluator.py` and
  `task_envs/spreadsheetbench/react_agent.py`.
- Project: SpreadsheetBench: Towards Challenging Real World Spreadsheet
  Manipulation, by Zeyao Ma, Bohan Zhang, Jing Zhang, Jifan Yu, Xiaokang
  Zhang, Xiaohan Zhang, Sijia Luo, Xi Wang, and Jie Tang.
- Source: https://github.com/RUCKBReasoning/SpreadsheetBench
- Data mirror used for the verified subset:
  https://huggingface.co/datasets/KAKA22/SpreadsheetBench
- License: Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/
- Changes: fixed train/validation/test selection, local path normalization,
  evaluator integration, and execution-agent adaptation for COBRA-Skills.

These derived files and benchmark assets remain subject to CC BY-SA 4.0; they
are not relicensed under the repository's Apache-2.0 license.

## SocialMaze

- Scope: `data/socialmaze_hard/`.
- Project: SocialMaze, by Zixiang Xu et al., Mohamed Bin Zayed University of
  Artificial Intelligence.
- Source: https://huggingface.co/datasets/MBZUAI/SocialMaze
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Changes: selected and reordered a fixed subset of the upstream `hard` split;
  added stable IDs, source indices, parsed labels, and split metadata.

## ALFWorld

- Scope: ALFWorld-originated assets under `data/alfworld/` and use of the
  separately installed `alfworld` package.
- Project: ALFWorld, by Mohit Shridhar and contributors.
- Source: https://github.com/alfworld/alfworld
- License: MIT. A copy is in `data/alfworld/LICENSE`.
- Changes: fixed experiment splits and added local manifests/configuration.

## SkillRL

- Scope: `task_envs/alfworld/vendor/`.
- Project: SkillRL.
- Source: https://github.com/NTU-LANTERN/SkillRL
- License: Apache License 2.0. A copy is the root `LICENSE` file.
- Changes: retained only the ALFWorld runtime subset, merged or trimmed helper
  modules, switched imports to the pip-installed ALFWorld package, and adapted
  prompt loading and environment integration for COBRA-Skills.

## SearchQA

- Distributed scope: ID-only files under `data/manifests/searchqa/`. Search
  contexts, questions, and answers are not distributed in this repository.
- Materialization source: https://huggingface.co/datasets/lucadiliello/searchqa
- Original SearchQA repository:
  https://github.com/nyu-dl/dl4ir-searchQA (BSD 3-Clause for that repository).
- License note: the Hugging Face mirror does not declare a dataset license, and
  SearchQA contexts contain third-party web snippets. Users materialize the
  selected records directly from the upstream mirror and must comply with its
  applicable terms.

## DocVQA

- Distributed scope: ID-only files under `data/manifests/docvqa/`. Questions,
  answers, and document images are not distributed in this repository.
- Materialization source: https://huggingface.co/datasets/lmms-lab/DocVQA
- Official benchmark: https://www.docvqa.org/datasets/docvqa
- Official data portal: https://rrc.cvc.uab.es/?ch=17
- License note: the Hugging Face mirror declares Apache-2.0. The official
  benchmark distributes document data through the registration-based RRC
  portal and identifies the document images as originating from UCSF. Users
  are responsible for the terms governing the materialized data.

## LiveMathematicianBench

- Distributed scope: ID-only files under `data/manifests/livemath/`. Theorems,
  proof sketches, questions, choices, and answers are not distributed here.
- Source: https://github.com/LinyangHe/LiveMathematicianBench
- License note: no license covering the monthly benchmark payload was located
  in the upstream release at the time this notice was prepared. Users must
  obtain the monthly files separately and supply them to the materializer.

Third-party names and marks are used only to identify their respective sources;
no endorsement is implied.
