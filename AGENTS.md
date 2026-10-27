# Tide Motion

Standalone native animation authoring tool. Keep Tide's engine and CLI independent.
The owner chose native executables and minimal user-installed dependencies. Do not
replace inference with Python, a hosted service, procedural animation or pose blending.

Read notes/implementation.md before changes. Record contracts, validation and gaps
there with the work. Keep model and dependency revisions pinned. Preserve upstream
licenses. Never distribute evaluation motion data or research-only checkpoints.

Run with TIDE_HIDDEN=1. Preview runs must be bounded with --frames or --capture.
Use only a Tide executable built from the engine checkout being validated.

Generated outputs are ordinary GLB animation assets. SOMA is the initial supported
rig; do not claim general retargeting or automatic skinning. Model arithmetic is
offline and is not Tide deterministic match simulation.

Use dev, validate locally, commit and push directly. No pull requests or additional
CI dispatches unless requested. Do not publish model files in Git.
