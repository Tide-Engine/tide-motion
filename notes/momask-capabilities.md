# Native MoMask authoring controls

MoMask is an opt-in native masked-token backend. Its exact manually supplied
checkpoint still lacks a verified author grant establishing commercial scope;
manual import and these features do not change that finding. See
[momask.md](momask.md) for primary release/license evidence and
[momask-local.md](momask-local.md) for the pinned bundle and installation.
No model, motion dataset, SMPL/SMPL-X asset or Python runtime is distributed.
Kimodo remains the default; ARDY, DART and other existing defaults are unchanged.

The pinned [transformer interfaces](https://github.com/EricGuo5513/momask-codes/blob/94a6636c9c463b7a9414c3401a6f1b67e6c51824/models/mask_transformer/transformer.py)
accept CLIP text, whole-body RVQ tokens, quantizer history and padding. There are
no learned Kimodo path/effector inputs or ARDY autoregressive history inputs.
Tide Motion implements token-context authoring around those real networks and
records any subsequent pose/spatial correction separately. It never replaces
model inference with procedural motion or blends two clips to generate a result.

| Tool | Implemented behavior |
|---|---|
| Text generation | ASCII prompts, seed, duration .2..600 s; native masked + residual sampling and RVQ decoding |
| Changing prompts | Repeated `--then "PROMPT" --duration SECONDS`, up to 16 segments / 600 s |
| History | `--history FILE.glb` generates a new-only continuation from observed source tokens |
| Append | `extend "PROMPT" --source FILE.glb --duration SECONDS` preserves the complete original prefix and appends native motion |
| Temporal inpainting | Sources up to 18,000 canonical frames; one `[begin,end)` or repeated ordered disjoint `--range BEGIN END` pairs, each at least 12 frames |
| Long edits | `--edit-mode continuous` (default) uses generated prefix context and a final observed source suffix; `anchored` retains source poses at internal window boundaries |
| Root locking | `--preserve-root` copies every original root transform for body-only inpainting; pose/path targets are refused with it |
| Full-body pose context | `--start-pose`, `--end-pose`, `--keyframe FRAME FILE[@F]`; observed whole-body RVQ tokens plus explicit correction |
| Mapped poses | `--pose-map BONES.json` uses existing explicit humanoid retargeting; GLB/FBX support follows that importer |
| Pose placement | `--placement auto|world|free`, `--follow world|ends`, `--end-at X,Z[,YAW]` |
| Root paths | Repeated `--path X,Z[,YAW]@FRAME`, optional `--dense-path`, `--path-margin 0..1`; native root warp/IK postprocessing |
| Hand/foot targets | `--effector PARTS FRAME FILE[@F]`; existing native MotionCorrection, measured residual errors |
| Pose selection | `--fit-poses --attempts 1..64` ranks worst full-body mean rotation error before correction |
| Footfall selection | `--footsteps 0..64 --attempts 1..64` filters geometric travelling footfalls; no matching candidate means no saved output |
| Loops | `--loop --in-place`: a second native endpoint-token pass, followed by explicit closure; endpoint equality is tested, velocity/contact equality is not guaranteed |
| Sampling | `--sampling categorical|gumbel|greedy`, `--residual-layers 0..5`, independent residual temperature/top-k threshold, `--unconditional` |
| Variants | `--variants 1..64`, sequential seeds, existing native contact sheets/GIF renderer |
| Contacts | `--contacts-out PLAN.json` exports qualifying predicted heel intervals for the separate existing cleanup command; no qualifying runs means no plan |
| CPU/GPU | `--threads 1..1024` (4), `--device cpu|auto|gpu` (cpu); Windows DirectML motion graphs, CPU text, strict GPU or auto restart/fallback |
| Rigs | SOMA-30, or `--rig soma77` presentation; inpaint/extend preserve the source rig and all its existing extra tracks |
| Other authoring | Existing mirror, retarget, cleanup, MotionBricks, preview and inspection commands consume these ordinary GLBs |

## Examples

Weights must already be privately imported. All commands use the normal native
executable; generation does not require the Tide renderer.

```sh
tide-motion generate "A person walks forward." --backend momask --duration 4 --then "A person waves with the right hand." --duration 4 --seed 7 -o sequence.glb
tide-motion extend "A person turns left while walking." --source sequence.glb --duration 3 --seed 3 -o extended.glb
tide-motion inpaint "A person walks forward and waves." --source extended.glb --range 30 90 --range 150 210 --preserve-root --seed 4 -o edited.glb
tide-motion generate "A person kicks forward with the right leg." --backend momask --duration 3 --start-pose idle.glb@0 --end-pose idle.glb@0 --fit-poses --attempts 4 --seed 1 -o kick.glb
tide-motion generate "A person walks forward." --backend momask --duration 3 --path 0,0@0 --path 1,2@88 --dense-path --seed 7 -o path.glb
tide-motion generate "A person walks forward." --backend momask --duration 3 --in-place --loop --seed 7 -o loop.glb
tide-motion batch examples/momask-authoring.json
```

`help momask` lists all defaults. Pose references accept `@first`, `@last` and
signed frames. Output constraints are canonical 30 Hz frame indices, including
negative indices from the end. One pose/effector source is allowed per frame;
combine effector parts into a comma list. No pose target may lie outside an edit's
strict interior or in an extension's preserved prefix. `correction none` leaves
pose tokens approximate and refuses paths/effectors; source endpoint/prefix
preservation and continuation root alignment still apply. `motion` uses the
existing MotionCorrection implementation (`kimodo` is an alias for that operation,
not a change of generator). Full-body pose roots use an explicit piecewise residual
warp before local pose correction. Paths target hips, not the smoothed-root track.

## Window, asset and preservation contracts

Each inference window stays within the model's 49 tokens / 196 samples at 20 Hz.
Short plain generation retains original rounding: 3 s requests 60 native samples,
exporting 89 samples at 30 Hz over the same interval. Timelines, history and long
generation request exact rounded 30 Hz output frame counts; extension includes the
source prefix, and the combined asset may not exceed 18,000 frames. Six canonical
samples correspond to four native samples / one RVQ token. A logical one-token
residual step uses a second masked padding slot: upstream's two condition-prefix
mask construction otherwise fails at length one. The output uses only the real
first token; reference tests cover this padding behavior.

`--history-frames 12..239` defaults to 119, token-aligned down; `--chunk-frames
12..293` defaults to 89 new frames. The native sampler may remask the last
`--context-overlap-tokens 0..4` (default 1) observed base tokens for context freedom.
`--preserve-context-residuals` additionally freezes residual code levels on known,
frozen context tokens. Observed prefix, suffix and pose fragments are encoded in
one shared heading frame. Full-body anchors use real five-frame constant-pose
observations; no interpolated placeholder clip is fed in as missing motion.
Unknown tokens are explicitly masked. Recipes retain the observed codes, token
mask, sampled codes, per-window prompt and independent RNG stream.

Continuation aligns the decoded root to the observed endpoint, applies the
configured local pose fade, restores observed frames and appends only newly
inferred frames. Continuous long edits use generated context with real lookahead;
the final window encodes the original source suffix separately. Anchored edits
instead use original source boundary poses at each internal window. Root endpoint
drift correction is linear over each edit interval. Neither mode guarantees
matching endpoint velocities or a globally smooth trajectory; the final source
poses and every outside frame are preserved explicitly.

SOMA-77 source extra tracks are preserved at every inpaint frame. Extension keeps
its entire prefix and holds the last original extra-joint pose in the new tail.
New SOMA-77 generations add static relaxed extra joints. No additional fingers or
facial motion are predicted. Custom rigs require explicit existing mapped-humanoid
retargeting. No automatic bone detection, skinning or non-humanoid claim is added.
GLB export may normalize/negate quaternion keys; preservation compares represented
poses with hemisphere matching, not file-byte identity.

Standard batch fields match the CLI in camelCase, including `prompts`, `history`,
`historyFrames`, `chunkFrames`, `contextOverlapTokens`, `preserveContextResiduals`,
`ranges`, `editMode`, `preserveRoot`, `startPose`, `endPose`, `keyframes`, `effectors`,
`path`, `endAt`, `poseMap`, `placement`, `follow`, `correction`, `correctionFrames`,
`densePath`, `pathMargin`, `fitPoses`, `footsteps`, `attempts`, `loop`, `contactsOut`
and all sampler controls. Pose/mapping input hashes and effective placement/follow/path
controls are retained; path yaw targets include measured heading residuals. See the executed [test manifest](../tests/momask-authoring.json)
and simpler [example](../examples/momask-authoring.json). The legacy `momask batch`
schema retains its earlier basic fields; use ordinary `batch` for the full tools.
All static inputs, options, seed ranges and output/recipe/contact-plan collisions
preflight before sessions. Geometry of references to earlier batch outputs is
checked once those outputs exist. Equal cache/device/thread jobs share sessions.
Auto GPU failure restarts the whole current search/job on CPU before saving.

## Validation and measured limits, 2026-10-28

ClangCL Windows x64 Release, RTX 4080 SUPER, isolated TEMP/TMP; GPU work and hidden
bounded previews hold `TideMotionGpuValidation`. All eighteen integrated model-free
CTests passed after rebasing concurrent equipment/review/DART changes. WSL GCC
warnings-as-errors syntax checks passed. All motion comes from own prompts/source
clips, without upstream evaluation data or SMPL/SMPL-X meshes.

A separate authored varying-finger source passes GPU edit and 6 s extension,
and an explicitly mapped custom humanoid pose passes generation/FK checks.
The 21-action native suite passes on CPU and GPU. It covers all tabled generation/
edit/sampler mechanisms, source SOMA-77 extension, full-body start/end/keyframes,
hand effectors, dense path, fit/footfall selection, loop and a 5-frame generation.
The CPU stress run additionally produces a valid 600 s / 18,000-frame asset.
Independent exported-track/FK tests check every outside/end pose, root-locked
transform, preserved prefix and extra track. Worst measured preserved component
roundoff is 5.96e-8 CPU / 8.94e-8 GPU. Failed 64-footfall searches save no output.

| Own-prompt case | CPU native seconds | GPU native seconds |
|---|---:|---:|
| 3 s plain walk | .151 | .342 |
| 8 s walk/wave timeline | .825 | 4.718 |
| 30 s continuation | 2.772 | 16.281 |
| Long edit [180,740) | 1.972 | 9.809 |
| 600 s continuation | 56.923 | Not run |

These are single local observations, including sampling/conversion and excluding
session construction; dynamic-shape GPU compilation contributes to cold cases.
The CPU stress batch elapsed 71.16 s, sampled process peak working set 630,239,232
bytes; the GPU suite elapsed 64.96 s, peak 724,979,712 bytes. Different work sets,
concurrent machine activity and sampled memory make these operational observations,
not a speedup or total-VRAM benchmark. CPU remains MoMask's default.

Independent official-tokenizer/CPU-ORT replay matches the native base and residual
choices for categorical, Gumbel, greedy, zero/partial residual, unconditional,
pose-token anchors, preserved residual context, multiple continuation streams,
long editing, timeline changes and both loop passes. Existing real PyTorch parity
at 4/7/49 tokens remains valid. Additional 12 real 1/2/3-token cases pass CPU ORT
bounds (1e-4 + 1e-3*abs(reference)) and native GPU bounds (.002 + .002*abs(reference));
RVQ encoder integer codes match exactly. Maximum new GPU absolute errors: masked
1.19e-5, residual 1.19e-5, decoder .0001297. Small-case sampled GPU-local process
peak is 326,729,728 bytes. CLIP stays the previously validated CPU tower.

Independent world-FK measurements on exported pose/keyframe assets give full-body
position residuals below 3.6e-7 m; the hand-effector example retains .01267 m error.
Dense path residual is .04004 m maximum to the smoothed targets; original requested
waypoints differ up to .05087 m because densification also smooths them. Recipes
report both. The tested loop closes every local transform with root peak speed
.6064 m/s, but derivative/contact continuity remains unproven.

HumanML-to-SOMA representation version 2 fixes an existing unstable roll hint on
bones whose incoming axis was parallel to it. It chooses a perpendicular basis
and uses mapped collars to resolve central torso yaw. The default walk's largest
adjacent local turn drops from about 119 degrees to 11.54 degrees, with exactly
unchanged model tokens and root translations. Analytic FK, normalization/RIC
roundtrips, shared-reference fragments and a small-feature-noise invariant pass.
This intentionally changes saved rotations relative to representation version 1.

Quality is not universal: measured adjacent local turns include 26.13 degrees in
the continuous long edit, 42.96 in extension, 68.01 in a short multi-range edit
and 71.32 in anchored editing. Endpoint equality does not remove model-induced
limb twists or establish smooth velocity. Foot sliding/penetration remains;
postprocessing and exact pose hits do not mean the network learned spatial
conditioning or that quality equals Kimodo. No new matched Kimodo quality ranking
is inferred from these tests. Existing comparison findings remain historical.

A current-checkout Tide build from `D:/Projects/PurrEngine` revision
`367ae4da2d31913176ae1e2624e1d78a1204f03c` captured and inspected an eight-clip
1920x2178 contact sheet using three hidden frames. Walking, kicking, waving,
spatial pose placement and the loop look plausible at sampled poses; sparse
images do not establish all-frame quality. No engine or CLI source was changed.

Reproduce developer checks (private pinned models required):

```powershell
pwsh -File tests/momask-authoring.ps1 -Tool build-momask/Release/tide-motion.exe -Models $env:LOCALAPPDATA/TideMotion/models -Device cpu -Stress
pwsh -File tests/momask-authoring.ps1 -Tool build-momask/Release/tide-motion.exe -Models $env:LOCALAPPDATA/TideMotion/models -Device gpu
python tests/momask-authoring.py PRIVATE_OUTPUT_FOLDER
python tests/momask-reference.py PRIVATE_SOURCE PRIVATE_BUNDLE PRIVATE_OUTPUT.glb
python tests/momask-small-shapes.py PRIVATE_SOURCE PRIVATE_CHECKPOINTS PRIVATE_BUNDLE NEW_PRIVATE_FIXTURES
# Hold TideMotionGpuValidation for this GPU-only manual test:
build-momask/Release/momask-gpu-tests.exe NEW_PRIVATE_FIXTURES PRIVATE_BUNDLE
```

The Python scripts are developer reference tools and are never installed. Runtime
models/revisions, bundle hashes, licensed dependencies and notices remain unchanged:
CPU ORT 1.31.1, isolated ORT DirectML 1.24.4 + DirectML 1.15.4 already used by ARDY.
See [ardy-native.md](ardy-native.md) for those exact runtime URLs/hashes/notices.
No new checkpoint/component grant was found or asserted by this implementation.
