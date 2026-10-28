# Agent test drive of advisory review

2026-10-28. Owner found manual interpretation of the calibration panel unhelpful
and requested an agent test drive. Baseline: c239748. The agent chose a real,
freshly generated idle/parry/idle MotionBricks asset, not a diagnostic mutation.
Private assets/recipes/measurements remain under ignored `output/agent-test-drive`.

## What happened

The original 51-frame clip has ten low-foot movement intervals. Its authored
guides already change stance, so movement cannot simply be treated as an error.
Toe centres meet the configured 6 cm reference-plane condition throughout all
50 adjacent sample pairs. Neither ankle has qualifying samples. This makes the
toe observations useful clues, not certification of planted-foot sliding.

The agent first tried native contact cleanup on the model's predicted heel runs:
left [0,15], [28,50]; right [0,9], [35,49], with protected frames 0,25,50 and the
ordinary 3-frame fade/.1 m correction bounds. All four intervals were rolled back:
three increased mean drift and one exceeded the correction bound. The exported
GLB independently matches unchanged root/translations, leg lengths and protected
poses. No guard was loosened to manufacture a successful repair.

Next the agent used real native MotionBricks inference, retaining both eight-token
segments, duration, source files, root placements, yaw and exact guide poses.
Three alternatives use target-poses 1,2,3 instead of the original held four-pose
conditioning. Target `hold` is removed, so the actual following source samples
supply whichever derivatives that control requests. Start context stays held.
The two-target batch reuses one native session set across all three alternatives.

| Candidate | Hint intervals | Left toe mean speed | Right toe mean speed | Max local joint step |
| --- | ---: | ---: | ---: | ---: |
| Original held four-pose context | 10 | 19.184 cm/s | 16.859 cm/s | 13.514 degrees |
| One target pose | 8 | 19.613 cm/s | 15.097 cm/s | 13.087 degrees |
| Two target poses (retained alternative) | 10 | 17.903 cm/s | 14.006 cm/s | 11.275 degrees |
| Three target poses | 9 | 18.164 cm/s | 14.686 cm/s | 12.901 degrees |

The one-pose result has fewer hints while worsening left-toe mean motion and
right-toe peak speed. Counting hints would therefore choose poorly. The retained
two-pose alternative reduces both toe means (6.7% / 16.9%) and peaks (42.453 ->
35.508 and 29.056 -> 26.238 cm/s), with the same 50 eligible pairs per toe. No
new ground/overlap proxy hints appear. All candidates retain frame count and
all local translations/quaternions at frames 0,25,50: independent dense-GLB checks
find zero position component difference and rotation difference below 1e-10
degrees. These constraints rule out winning by omitting the guide action.

This is a limited, single-clip measured improvement, not an overall quality winner
or proof of correct contacts. The original and every attempted candidate/recipe
are retained. No automatic acceptance, threshold changes or style restrictions.

## Tool correction prompted by actual use

The first report localized events but required extracting display tracks from the
HTML to compare continuous movement. Native review now emits `measurements` for
every evaluated rule/subject, even without findings: eligible samples, threshold
exceedances, mean, maximum and peak frame. Zero qualifying samples yields null,
not zero movement. Quaternion statistics exclude frame zero. Existing observations
and feedback identities are unchanged; these are additive schema-1 fields.

Independent analytic tests cover two 90-degree jumps over seven adjacent pairs,
their mean, exact sample/exceedance counts and unavailable low-foot measurements.
Native review contracts and CLI contracts pass. No inference algorithm, model,
dependency, pin, license or correction behavior changed.

## Motion inspection and reproduction

The ignored `agent-decision.json` records the agent's interpretation, all attempts,
selected recipe, invariant checks, measurements and limits. `measured.html.json`
contains native per-rule measurements for all five compared clips. The private
`before-after.gif` plays original and alternative together; `poses.png` contains
six selected moments per clip in front/profile. The pose selection is per clip,
not synchronized; the GIF uses matching frame counts for simultaneous playback.
Sparse inspection shows the same arm-raise/recovery pattern and no gross binding
failure, but does not certify full motion or mesh/equipment interaction.

The first animated comparison exposed another presentation issue: each narrow
GIF cell carried the still sheet's full metrics label, spilling into the next
candidate. Animated cells now show their short candidate label; the wide still
sheet retains detailed metrics. A new 51-frame hidden capture verifies readable
Original/Alternative labels. The final GIF places original left, alternative right.

Integration preserves dev f08cb0e's equipment authoring work. The combined Windows
Release CPU build passes all 18 CTests; the final label-only adjustment passes
the native preview contracts and bounded engine capture. Model arithmetic and
the measured candidate assets above are unchanged by integration.

Renderer: engine checkout D:/Projects/PurrEngine, clean 367ae4da2, targets tide_cli
and tide_platform rebuilt and privately staged. Staged CLI matches the checkout
build SHA-256 7ba1c9d342bddbd5671534efd3b81e6f5d1f28fa4913866b68ae71f66a98907a.
TIDE_HIDDEN=1 and TideMotionGpuValidation held for both bounded captures. GIF
records 51 frames at 720x346; PNG is 960x1088. No owner installation or engine
source changes. Weights and paid/derived animations remain outside Git.

To reproduce from a local calibration set, clone its guides JSON into three
manifests, resolve input paths, remove `hold` from targets and set target `poses`
to 1/2/3. Keep start context, source pose frames, placements and token counts.
Run ordinary native `pose-sequence` actions in one batch, then `review` with the
original, cleanup attempt and three alternatives at identical review settings.
Verify dense GLB key poses before interpreting the continuous measurements.

Rebased again onto dev 66ef74b, preserving 338c4e1 DART controls and subsequent
MoMask authoring changes, including their model pins/notices. Appended-note conflict
was resolved by retaining both complete histories. Latest combined Windows Release
CPU build passes all 18 CTests. Running native review again on all five retained
clips exactly reproduces every observation and continuous measurement. No new
inference/benchmark is claimed for unrelated backend changes; MotionBricks assets
and their independent key-pose checks remain the same.

## Owner feedback after playback

The owner preferred the original left toe and agreed the observations captured a
useful difference. This supersedes any inference that the lower-motion alternative
was aesthetically preferable on that side. Retain both assets; no global clip
winner or preference about the right toe is inferred. The feedback names a body
part but no timestamp, so do not invent a particular problematic frame or classify
every left-toe interval as an error. The private agent decision now records that
scoped human preference separately from the unchanged measurements.

The owner authorized proceeding with the hints as advisory agent feedback, without
requiring a human calibration exercise. AGENTS.md now makes that the authoring
workflow and explicitly rules out automatic correction, export gates and metric-
based artistic ranking. No detector, inference, threshold or runtime behavior
changes were needed for this clarification.
