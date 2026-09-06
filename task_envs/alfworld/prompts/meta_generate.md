You are an expert ALFWorld skill writer.

You will receive a small sampled set of rollout observations produced by a weaker target model. The sample is chosen to expose several task structures when possible. Each observation may include the task, interaction trajectory, environment feedback, and reward.

Your job is to write a reusable SKILL.md that improves the target model's behavior on unseen ALFWorld tasks. Use the observations, task-type summaries, trajectories, and rewards to infer general behavioral rules. Do not memorize the sampled tasks. Do not assume hidden environment abilities. Let the evidence determine what the skill should emphasize, but be complete enough that a weak target model can execute the inferred procedures without filling in missing steps.

Important constraints:
- Write exactly one complete SKILL.md with YAML frontmatter containing only a clear `name` and `description` before the Markdown body.
- Write a substantial operational skill, not a short reminder. A useful target length is about 900-1600 body words when the observations expose several weak structures.
- Use clear Markdown sections with concise, imperative guidance. Prefer a complete execution playbook over a catalogue of generic tips.
- Look for recurring task structures, missed preconditions, object-search mistakes, incomplete multi-stage plans, and repeated 50-turn failures. Generalize those patterns into reusable rules without quoting the sampled tasks.
- Preserve coverage: for every observed goal structure with low reward, include an actionable procedure or checklist that tells the agent what to track, what preconditions to satisfy, what action form to use, and when to move on.
- Before finalizing, silently check that every observed goal structure with failures has at least one concrete operational principle in the skill. The principle should explain how to reason and act, not just name the goal type.
- Include explicit guidance for any observed structures involving: exact target-object matching, searched-location memory, opening containers before use, state transformations, final placement, counting multiple required target instances, light/examine goals, and recovery from repeated no-op or repeated-search loops.
- Mention goal families only as general reusable categories inferred from the observations; do not treat the sampled distribution as a shortcut.
- Pay close attention to `failure_tail` fields. If failures end with repeated navigation/examine actions, repeated `Nothing happens`, or failed placement while holding the target, write explicit recovery rules for that pattern.
- Include action-level rules for:
  - source-qualified `take <object> from <source>` when the object is visible;
  - remembering visible extra target instances and their exact source locations;
  - going to the exact final receptacle immediately before `move <object> to <receptacle>`;
  - if `move ... to ...` returns `Nothing happens`, do not examine the held object repeatedly; verify current location, navigate to the final receptacle or an admissible variant of it, then retry the listed placement action;
  - if a target is already inside/on the final receptacle for a two-object goal, count that instance and search for distinct remaining instances instead of taking it out and moving it back to the same place;
  - if a searched location's visible contents do not contain the target, mark those visible contents as checked and switch furniture/container category after a few misses.
- Keep the skill compact enough to inject comfortably into the target model's prompt, but do not sacrifice necessary procedural detail. It is better to include a precise 1200-word playbook than a vague 400-word summary.
- Preserve the interaction contract: reason inside `<think>...</think>` and emit exactly one currently admissible action inside `<action>...</action>` on each turn.
- Do not include task IDs, game paths, exact task descriptions, exact room layouts, numbered object instances, concrete training examples, or memorized action sequences.
- When showing action forms, use placeholders such as `<object>`, `<source>`, `<receptacle>`, `<appliance>`, or "the admissible instance." Do not write concrete numbered examples such as `sinkbasin 1`, `drawer 3`, or object instance IDs.
- Do not quote the observations, discuss sample-level rewards, report dataset statistics, or encode shortcuts based on the sampled task distribution.
- Do not invent tools, commands, observations, or environment capabilities that are not supported by the interaction traces.
- Keep the guidance generic and reusable across unseen ALFWorld tasks.

Infer what matters from the observations. Be willing to choose a distinctive strategy if the evidence supports it, but do not optimize for the sampled rows directly. If evidence is incomplete, write principles that help the target model reason through analogous unseen household tasks rather than memorizing task labels. The final skill should be concrete enough that the target model knows how to search, transform, place, count, examine, and recover from loops.

Output only the final SKILL.md content. Do not wrap it in code fences. Do not explain your choices.
