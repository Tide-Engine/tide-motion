# Tide Motion

Standalone native animation authoring tool. Keep Tide's engine and CLI independent.
The owner chose native executables and minimal user-installed dependencies. Do not
replace inference with Python, a hosted service, procedural animation or pose blending.

Read notes/implementation.md before changes. Record contracts, validation and gaps
there with the work. Keep model and dependency revisions pinned. Preserve upstream
licenses. Never distribute evaluation motion data or research-only checkpoints.

Run with TIDE_HIDDEN=1. Preview runs must be bounded with --frames or --capture.
Use only a Tide executable built from the engine checkout being validated.

Generated outputs are ordinary GLB animation assets. Models use SOMA; mapped
humanoids can be retargeted offline in either direction. Do not claim automatic
bone identification, skinning or non-humanoid retargeting. Model arithmetic is
offline and is not Tide deterministic match simulation.

For Kimodo authoring, choose the checkpoint by action using
PROMPTING.md#choose-the-kimodo-checkpoint-by-action and its linked comparison.
Set --model explicitly in agent-generated commands and model in batch actions.
Start with v1.2 for longer plain right kicks or path accuracy, v1.1 for backward
walk/turn foot contacts or the tested sword-worded kick. These are measured
starting points: compare matched prompts, durations, seeds and constraints when
quality matters, inspect the motion, and retain the chosen asset and recipe.
v1.1 remains the compatibility default; do not infer quality from version order
or claim automatic action routing. This guidance does not rank the ARDY backend.

Use dev, validate locally, commit and push directly. No pull requests or additional
CI dispatches unless requested. Do not publish model files in Git.
