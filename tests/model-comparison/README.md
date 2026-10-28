# Paired SOMA-RP v1.1/v1.2 comparison

Developer-only evaluation tooling. Generation is the real native Tide Motion
executable; Python only creates manifests and independently reads exported GLBs.
No evaluation dataset, checkpoint or generated motion belongs in Git.

From a built Vulkan checkout, with both pinned models installed and a locally
generated SOMA idle clip of your own:

```powershell
python tests/model-comparison/protocol.py .research/compare OWN-IDLE.glb
./tests/model-comparison/run.ps1 -Tool build/Release/tide-motion.exe -Models MODELS -Work .research/compare
python tests/model-comparison/score.py .research/compare
python tests/model-comparison/test_metrics.py
```

The fixed protocol contains 400 clips / 200 pairs, seeds 101..108, default
separated CFG 2/2, 50 steps, explicit GPU. Ten plain self-authored prompts each
use 2.5 and 5 seconds. Additional pairs cover 2.5/4-second kicks constrained to
the same input start/end poses, a three-waypoint curved path, walk->wave and
right-kick->left-kick sequences. No cleanup or in-place travel removal is applied.
The four 100-clip blocks use v1.1 / v1.2 for seeds 101..104, then v1.2 / v1.1 for
105..108. One Windows named mutex TideMotionGpuValidation guards all inference;
the runner samples host working set and flags any other tide-motion process.
Contention and profiling/logging overhead must be disclosed for performance.

The scorer checks all expected frame counts, finite FK positions and model
identity recipes. Proxies were defined before comparing results:

- walk/backward: >0.5 m root displacement in the named initial-facing direction;
  run: >0.75 m plus at least three frames with both feet/toes above 8 cm.
- kick: named ankle >45 cm and >15 cm above the opposite ankle's peak; the
  sequence checks right in the first segment and left in the second.
- jump: both feet/toes above 8 cm for >=3 frames, root height range >12 cm, and
  feet back down throughout the final five frames.
- crouch transition: head height range >40 cm and final five-frame mean >30 cm
  below the first five-frame mean. A crouched clip starting low fails this
  transition proxy even if its crouched pose is plausible.
- wave: right hand rises above the head and at least 10 cm above the opposite
  hand's peak relative to the head. This checks raising the named hand, not the
  number/shape of waves. The sequence also requires >30 cm first-segment travel.
- turn around: >135 degrees net heading change from the first frame's hip line.

Foot slide is horizontal cm/frame for ankle/toe samples when both adjacent
heights are below 3 cm. Report contact sample fraction alongside it: lifted feet
or omitted motion can otherwise win this metric. Both-success-only comparisons
are also included. Foot/toe joint positions below -2 cm yield a penetration-frame
proxy; joints are not the mannequin's sole surfaces. Local quaternion step max
and p95 flag abruptness; rapid intended motion can legitimately score higher.
Recipe modelError measures pose error before correction; path error measures
hips against waypoints (the model conditions the smooth root).

Summaries retain cases separately. Paired mean-difference intervals resample the
eight seed clusters, averaging durations within each seed, with a fixed bootstrap
RNG and 2,000 resamples. These small-sample descriptive intervals are not
multiple-comparison-adjusted significance claims. Proxy success cannot replace
visual assessment of semantics, balance, naturalness, contacts or timing.

For blinded visual review, build the optional uninstalled helper:

```powershell
cmake --build build --config Release --target soma-sheet
build/Release/soma-sheet.exe PAIR.json CHECKOUT-BUILT-TIDE OUTPUT.png
```

PAIR.json is an array of two {"file":"absolute clip.glb","label":"A"} entries.
A .gif output instead records the existing bounded native animation preview.
The helper renders the existing native six-key-pose contact sheet, two views per
clip, through bounded hidden Tide. Hold TideMotionGpuValidation for these renders,
set TIDE_HIDDEN=1 and use Tide built/staged from the engine checkout being checked.
Keep label-to-model keys hidden until recording votes. Key poses are selected per
clip and are not synchronized; this is sparse pose review, not a full-motion or
human panel study. The optional helper is excluded from ordinary builds and from
runtime installation.
