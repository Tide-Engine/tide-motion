# SOMA-RP v1.1 versus v1.2: broader native comparison

2026-10-28. This is a local comparison on self-authored prompts and generated
poses, not an upstream benchmark or a human preference study. No evaluation
motion data was downloaded, used or distributed. Model revisions, conversion
hashes, commercial licenses and native dependency pins remain unchanged.

The native Vulkan executable was built from this checkout at 15c78fd. Generation
uses 50 steps, text/constraint guidance 2/2, seeds 101..108, no in-place removal
and no contact cleanup. Ten actions each have 2.5- and 5-second clips. Additional
pairs cover start/end-constrained kicks at 2.5/4 seconds, a curved three-point
path, walk->wave, and right-kick->left-kick sequences. There are 400 GLBs / 200
matched pairs. Four blocks alternate v1.1/v1.2 for seeds 101..104, then v1.2/v1.1
for seeds 105..108. Every generation uses the native backend, never Python.

The common pose reference is an existing locally generated v1.1 idle; it is held
constant for both versions, but it does not establish symmetric performance on
all possible reference poses. Both model/recipe identities, expected frame
counts and finite exported FK values are checked. The developer-only scorer
reads the ordinary GLBs independently of native inference. See
tests/model-comparison/README.md for all thresholds and reproduction commands.

Success counts are explicitly geometry proxies: displacement/direction, the
named foot rising, flight plus landing, a standing-to-crouch height change, the
named hand rising, or heading change. They do not establish semantic precision,
balance, graceful motion, a particular number of footfalls, or correct hand
waving. Contact sample fraction accompanies conditional foot slide; comparisons
where both versions passed the action proxy are also retained. Joint positions
below the floor are not sole-surface measurements. Large quaternion steps can
also be intentional fast movements.

The paired difference intervals resample whole seed clusters, averaging durations
within each seed, with 2,000 fixed-RNG bootstrap samples. They are descriptive
small-sample intervals without multiple-comparison correction. No frame is
treated as an independent trial. Twelve visual pairs were selected before the
comparison: seed 101, short duration for each plain action and both sequences.
Their labels are randomized A/B with a hidden key until votes are recorded.
The native six-key-pose sheets are sparse visual checks, not full temporal review.

All four native blocks exited successfully under TideMotionGpuValidation. Each
block observed another tide-motion process at least once; that flag does not
identify its workload or prove it used the GPU. Isolated performance was not
established. A brief helper build and CTest run also overlapped generation.
Consequently these runs support quality comparison, not a speed winner. After
generation released the mutex, a quiet window was left for the concurrent ARDY
measurement before any comparison rendering began. Generated clips, logs,
scores, reference-pose hash, executable hash and blind keys stay ignored beneath
.research/broad-v12; no model or generated motion is added to Git.


## Results

All 400 expected GLBs passed frame-count, model-identity and finite-FK checks.
Across the 192 pairs with an action proxy (path pairs are excluded), v1.1 passes
164/192 (85.4%) and v1.2 167/192 (87.0%). The paired seed-cluster bootstrap interval
for v1.2 minus v1.1 is -2.60 to +5.73 percentage points around +1.56. This does not
establish an overall winner, and the count is not a measured semantic accuracy.

| Action | Pairs | v1.1 proxy passes | v1.2 proxy passes |
| --- | ---: | ---: | ---: |
| backward | 16 | 16 | 16 |
| crouch | 16 | 16 | 16 |
| jump | 16 | 15 | 15 |
| kicksequence | 8 | 7 | 8 |
| leftkick | 16 | 15 | 16 |
| path | 8 | n/a | n/a |
| posekick | 16 | 12 | 13 |
| rightkick | 16 | 9 | 14 |
| run | 16 | 15 | 16 |
| swordkick | 16 | 9 | 7 |
| turn | 16 | 16 | 16 |
| walk | 16 | 16 | 16 |
| walkwave | 8 | 5 | 3 |
| wave | 16 | 13 | 11 |

The clearest prompt-specific action difference is the plain right kick: both
versions pass 8/8 short clips, but at 5 seconds v1.1 passes 1/8 and v1.2 6/8.
This is five v1.2-only successes and no v1.1-only successes across the 16 pairs.
It does not transfer to the sword prompt: short clips pass 8/8 versus 6/8, long
clips 1/8 for each. The sequence right->left passes 7/8 versus 8/8. These small
counts do not prove performance on other prompts, durations, seeds or guidance.

### Contacts, constraints and abruptness

Mean slide is cm/frame, averaged per clip and then across pairs. The intervals
below describe paired v1.2-minus-v1.1 differences, resampling seed clusters.
All backward, turn and crouch clips pass their action proxies in both versions.

| Case | v1.1 slide | v1.2 slide | Difference interval, 95% |
| --- | ---: | ---: | --- |
| backward walk | 0.182 | 0.494 | +0.127 to +0.652 |
| turn around | 0.100 | 0.200 | +0.027 to +0.185 |
| crouch transition | 0.196 | 0.109 | -0.165 to -0.017 |
| forward walk | 0.199 | 0.181 | -0.082 to +0.019 |
| run | 0.520 | 0.422 | -0.288 to +0.049 |

Both-success-only comparisons preserve the mixed direction: right kicks
0.164->0.186 (9 pairs), pose kicks 0.191->0.239 (11), sword kicks 0.177->0.204
(7), walk->wave 0.157->0.181 (3), runs 0.530->0.434 (15). v1.1 and v1.2 have
35.5% and 34.7% near-ground adjacent foot/toe samples overall. These conditional
metrics omit raised feet and do not replace contact inspection.

Raw pre-correction pose-kick mean joint-position error is 1.80 cm versus 2.15 cm;
paired difference interval +0.015..+0.676 cm. The common pose comes from v1.1, so
this does not establish symmetric reference-pose superiority. Both versions'
full-body correction remains exact. Curved-path mean hips-waypoint error is
6.61 versus 5.31 cm; difference interval -2.05..-0.63 cm. This measures hips,
not the conditioned smooth-root feature. Walk->wave start-pose error is nearly
equal, 1.46 versus 1.43 cm.

There is one clip per version with a local joint step above 60 degrees in one
30 Hz frame (v1.1 wave/5s/seed107; v1.2 walkwave/seed102). The v1.2 backward walk
has a larger mean per-clip maximum step, 18.53 versus 13.81 degrees, but a fast
intentional kick also raises this metric. Mean fraction of frames with any
foot/toe joint below -2 cm is 0.05% versus 0%; these are joint proxies, not mesh
penetration measurements.

### Blinded sparse visual review

Twelve A/B pose sheets were rendered through Tide built from
D:/Projects/PurrEngine at 28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c, after
rechecking its native CLI and MinGW platform targets. The prior private staging
has the matching binary/engine resources. TIDE_HIDDEN=1 and bounded three-frame
captures were used. Votes were written before opening the randomized model key.
This is one assistant's limited pose-sheet review, not a blinded human panel or
full-motion rating.

- v1.2 preferred: plain right kick (clearer extension), crouch (clear squat
  rather than kneeling), turn (less unrequested overhead arm flourish).
- v1.1 preferred: backward walk (upright walking instead of a hunched, rapid
  retreat), sword-prompt kick (clearer kick).
- Seven ties: walk, run, left kick, jump, wave and both sequences.

Both versions show recognizable motions in these sampled frames without broken
binding or inverted limbs. Each sheet uses its clip's own six selected key poses;
frames are not synchronized. A requested optional GIF render correctly refused
because this exact checkout-built Tide lacks `help run --record`. No GIF or full
continuous-motion visual validation is claimed; the twelve PNG sheets succeeded.

### Performance limits and decision

Four 100-action block wall times were 180.78, 209.43, 182.71 and 162.00 seconds
(v1.1, v1.2, v1.2, v1.1). Their sampled peak host working sets were 0.97, 0.41,
1.04 and 1.03 GiB. These include process loading/hashing and depend on OS paging;
they are not VRAM or per-model tensor-memory measurements. Other-process flags,
profiling overhead, sequential scheduling and a brief developer build preclude
claiming a speed or memory winner. No other process was stopped or reconfigured.

Keep v1.1 as the compatibility default. Prefer trying v1.2 for longer plain
right kicks and paths; v1.1 remains useful for backward walking/turning contacts
and the tested sword prompt. There is no broad quality dominance justifying a
default change from these results. Both stay selectable. Corrected training/test
splits in v1.2 are still preferable for clean upstream benchmarking, independently
of these locally observed animation tradeoffs.
