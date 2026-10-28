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

Write conventional commit messages (`feat:`, `fix:`, `perf:`, `docs:`, `ci:`,
`test:`, `refactor:`, `chore:`; `!` after the type for a breaking change). A push
to dev whose commits since the last release have a feat or fix publishes the
next version once every platform passes (`feat` the minor, `fix`/`perf` the
patch, breaking the minor below 1.0); other types publish nothing. The version
is worked out by `.github/scripts/next-version.sh`; don't edit it in CMakeLists.

For motion authoring/revisions, use `review` and its adjacent JSON as advisory
agent feedback; see notes/advisory-review.md. Keep reports with the retained clips
and recipes in ignored output. A human rating exercise is not required. Hints
must not block export, automatically trigger correction, or rank artistic quality
by their count or by lower measurements. Human feedback and intended action/style
take precedence; preserve originals and intentional deviations. The owner preferred
the original left toe in the parry test despite the alternative's lower toe-motion
measurements (notes/review-test-drive.md). Treat that as scoped feedback, not a
universal preference for more or less movement. Unchecked properties stay unknown.
