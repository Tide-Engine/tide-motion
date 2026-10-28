# Native DART locomotion backend

Implemented 2026-10-28. `dart PLAN.json -o CLIP.glb` is an explicit opt-in CPU
backend for continuation and waypoint locomotion, with the released policy's
`walk`, `run`, `hop on left leg` labels. It is offline animation authoring.
It does not expose arbitrary text, imported start clips, gradient inbetweening,
scene/SDF interaction, automatic rig identification, skinning or non-humanoid
retargeting. Existing backends and defaults are preserved.

## Exact assets, rights and preparation

All download URLs, byte sizes and SHA-256 values are pinned in
[dart-models.json](dart-models.json) and the native registry.
DART source is pinned to `bb67ae1ed6ce051080468bf15bc6e54a6c3f8417`.
The [author's checkpoint statement](https://github.com/zkf1997/DART/issues/25#issuecomment-3806649923)
explicitly grants Apache-2.0 for code/checkpoints, retaining other projects'
terms. This is the basis for using the released inference models, including
their necessary normalizer, after the owner's clarification. No separate
normalizer prohibition was found; its earlier classification as a blocker was
too broad. This does not relicense AMASS training data or SMPL assets.

The installed set is the euler denoiser `checkpoint_300000.pt`, VAE
`checkpoint_200000.pt`, locomotion policy `iter_2000.pth`,
`mean_std_h2_f8.pkl` and official OpenAI CLIP ViT-B/32. CLIP source is pinned to
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (MIT), with the official checkpoint
digest verified. DART and CLIP notices ship in `licenses/`. No SMPL-X/SMPL-H
parametric model, rest-body extract, sample `stand.pkl`, motion dataset or
evaluation asset is used or downloaded by this backend. The normalizer is an
inference coefficient file, not a redistributed collection of training motions.
No training/fine-tuning was performed, and nobody was contacted.

`models install dart --accept-licenses` downloads these five exact files and
builds four ONNX graphs plus one F32 tensor file locally in native C++. It never
executes Python, TorchScript or a pickle interpreter. Its tensor byte ranges and
F16-to-F32 conversion apply only after full source size/hash verification.
The 372,168,788-byte tensor file must match SHA-256
`6c186ea7807efe7cef02e639726bb94040f1c44fb8beef0621c667440d7b7877`
before rename; each graph has its own pinned hash. Existing good files remain
cached, partial downloads resumable. Every first model load verifies the set.
`all` keeps its existing selection and does not acquire DART.

`src/generated/dart-graphs.inc` is readable generated C++ operator/schema source with external
tensor locations and source byte ranges, not checkpoint tensors. Every learned
initializer is external; large embedded constants are checked to be mathematical
zeros/causal attention masks. No ONNX/checkpoint/normalizer/tensor file enters
Git or runtime archives. Raw sources plus converted cache require about 875 MB.

The development-only graph compiler is `tools/dart_export.py`; its environment
is pinned in `tools/dart-export-requirements.txt`. It verifies source/checkpoint
pins before parsing, loads checkpoint tensors with `weights_only=True`, and uses
a restricted statistics unpickler. It compares exported tensors with PyTorch,
emits a local reference fixture and generates the native schema source. Neither
this tooling nor its environment is packaged or invoked by installation or
inference. Reproduce with `--source PATH_TO_PINNED_DART --clip-source PATH_TO_PINNED_CLIP_DIRECTORY
--models PATH_TO_VERIFIED_RAW_FILES --emit src/generated/dart-graphs.inc`.

## Body-free host and conversion contracts

The fixed forward graphs are CLIP text, transformer denoiser, autoencoder decoder
and policy actor/distribution. The VAE latent is `[1,1,256]`, history `[1,2,276]`,
future `[1,8,276]`; denoiser width 512/eight layers, decoder width 256/seven skip
layers. The actual opset-17 graphs have 1,277 / 823 / 84 / 1,894 nodes respectively
for denoiser / decoder / policy / text, all standard ONNX operators, with the full
counted inventory in the pin manifest. `latent_std` must be replaced with its checkpoint F32 buffer after loading:
upstream initializes an integer buffer, so plain `load_state_dict` truncates its
0.874... scale to zero. The exporter explicitly follows upstream's replacement;
native tensor parity covers the corrected decoder.

The sampler is x0 prediction, ten cosine-schedule DDIM steps, eta zero, CFG 5,
with two denoiser branch evaluations per step. Policy observations are the
unnormalized 1069-value goal/text/history/scene vector, with 60-degree heading
and 5 m distance clipping. The actor's tanh-scaled mean and clamped exponential
standard deviation produce sampled initial latent noise. It is not replaced
with the mean. Native RNG is mt19937 plus the compiler library's normal transform;
an equal seed is not a promise of PyTorch or cross-platform identity.

Initialization starts from normalized zero (the model mean), followed by 1..32
native walk primitives (default eight). No procedural animation is substituted
for inference. At fixed shape, pelvis position equals translation plus the
constant rest pelvis offset. Linearity gives that offset from the normalizer's
`mean(joint0)-mean(transl)`:
`[0.0009881773, -0.3752664328, 0.0128494073]` metres. This avoids reading or
extracting values from a separately licensed body model. It is subject to F32
statistics rounding; exact agreement with upstream body-model calibration has
not been measured and is not claimed.

Each primitive carries the last two frames, reconstructs the hip-line frame,
reframes all redundant positions/deltas, conjugates orientation deltas, updates
the world transform and normalizes again. Native row-6D decoding is separate
from MotionBricks' column convention. World reconstruction preserves the pelvis
offset term. Targets advance on 0.3 m horizontal pelvis error or budget exhaustion;
misses are exported with explicit recipe metrics, never silently corrected.

The two history frames are emitted once, then eight frames per primitive at
30 Hz, capped below 18,000 total frames. All segments, labels, target dimensions,
finite ranges, counts, output collisions and batch plans are checked before
model loading. A RunContext owns one set of CPU sessions and three CLIP embeddings;
another threads setting reloads it. No daemon or service exists.

The fixed anatomical index/name map is DART's known 22-joint topology to SOMA.
A source reference's fixed offsets are recovered from the first generated
pose's positions and parent rotations. This reconstructs generated geometry;
it does not identify arbitrary bones or load a parametric body. Source root
rotation is global, body rotations local. The proper Z-up to Y-up basis maps
`(x,y,z)` to `(-x,z,y)`; child rotations stay in their original local axes.
Existing explicit humanoid calibration/retargeting converts to canonical SOMA.
Root scale is one; horizontal root coordinates are preserved exactly so target
errors describe the exported asset. Non-root translations are the fixed SOMA
offsets, missing twist/end slots inherit their documented unmapped reference
transforms, and quaternions are normalized. No contact IK or pose blending is
part of this backend. The first generated reference is approximate, and
predicted-joint/rotation inconsistency can affect contacts after conversion.

## Validation actually completed

Windows x64 ClangCL 18.1.8 Release, ONNX Runtime 1.31.1 CPU, four threads:

* Four real native graphs compared against independent exported PyTorch tensor
  fixtures: maximum absolute error `7.03335e-6` (test limit `2e-4`). This covers
  denoiser, corrected decoder, policy mean/std and CLIP embedding; it is not
  full upstream SMPL rollout parity. `dart-tests MODELS/dart reference.json`
  reproduces the opt-in comparison.
* Native conversion produced byte-identical pinned graphs/F32 tensors. A
  separate cache fetched the missing normalizer over HTTPS, verified it, and
  converted the models successfully. Full model transfers used curl before
  integration rather than downloading the same 500 MB twice.
* Twelve model-free CTests pass after integrating the concurrent ARDY, HumanML and
  transactional upgrade work, including DART ranges/labels/unknown fields,
  budget and malformed target refusals, CLI help/license gating, and a late bad
  DART batch plan failing before an earlier inference action loads models.
  TEMP/TMP are worktree-local. No existing generator algorithm was changed.
* Ubuntu 24.04 x64/GCC 13 under WSL also builds with project warnings as errors
  and passes all twelve CTests. The four native graph/reference comparisons have
  the same maximum `7.03335e-6` error. A real Linux three-waypoint run reaches
  all three goals (approximately 0.27, 0.29 and 0.07 m error), with a valid
  146-frame GLB. Normal sampling differs from the Windows compiler library, as
  documented. macOS/ARM64 inference has not been run locally.
* `tests/dart-inference.ps1` ran 24 native actions with one verification/load:
  eight seeds each of three walking waypoints, changing walk/run/hop waypoints,
  and 12+12+12-chunk text continuation. All **48/48 waypoint arrivals** were
  within 0.3 m. Independent GLB decoding/FK verified recipe goal errors to
  `1e-4` m and maximum bone-length variation below `2.7e-7` m. Mean/max target
  error was **0.1484 / 0.2981 m**. These are own prompts/model-generated warmup,
  not dataset or upstream benchmark results.
* Run segments were faster than walk segments in all eight continuation clips;
  mean root speeds were 1.19 m/s walking, 2.84 running, 1.26 hopping. These are
  geometric adherence proxies, not a semantic benchmark. One hopping clip
  lifted the requested support leg higher than the other; support-side fidelity
  is not guaranteed.
* Largest joint step was 22.39 degrees. Largest measured segment-boundary root
  step was 0.1413 m and velocity change 0.6217 m/s. Foot-slide proxy (feet/toes
  below 3 cm on consecutive frames) averaged **3.22 cm/frame**, max clip mean
  5.87 cm/frame. This is substantial sliding, includes imperfect contacts and
  differs from upstream metrics. No cleanup was applied or improvement claimed.
* Observed 24-action process time 30.62 s, sampled peak working set 321,949,696
  bytes (100 ms sampling; may miss brief peaks). Median per-action host generation
  time was 1.01 s, including warmup/conversion, excluding initial verification
  and session loading. These are machine observations, not isolated GPU benchmarks.
* A 3,074-frame / 102.43 s walk→run→hop continuation generated in 14.19 s and
  reimported through the strict native loader. Independent FK length error was
  below `2.8e-7` m; slide proxy worsened to 6.22 cm/frame. Long output is functional,
  not a claim of sustained contact quality.
* Bounded hidden 960×540 mannequin capture succeeded and was inspected, using
  Tide built from clean D:/Projects/PurrEngine revision
  `28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c`, staged privately by the ARDY task.
  `--frames 51`, TIDE_HIDDEN=1, no binding diagnostics or inverted limbs in that
  sampled pose. No engine source changed. No GPU inference ran, so there was
  no GPU workload requiring the shared TideMotionGpuValidation mutex.
* The Windows runtime archive passes installer/reinstaller, checksum and traversal
  refusal checks in an isolated installation with spaces/apostrophes. The installed
  executable generates both example plans and strictly reimports their 146/290-frame
  canonical SOMA GLBs with PATH restricted to Windows system directories; the two
  actions share one session load. DART examples, notices and pin manifest are
  installed. No Python files, checkpoints or ONNX model files are in the archive.
  A separate raw-only cache reconstructs all four graphs and the tensor file with
  the readable native graph builder, matching every pinned digest.
* `dart-tests --assets DIR` checks canonical clips and native 60 Hz half-frame
  sampling against independent double-precision linear/quaternion midpoint
  calculations. The 24 generated assets supply 4,624 midpoint samples: maximum
  position error `9.53674e-7` m and quaternion dot error `8.62042e-8`, identical
  on Windows and Linux. DART's model/export rate remains 30 Hz; there is no
  unvalidated 20 Hz checkpoint conversion.
* Final rebase onto `24f17c9` preserves ARDY automatic GPU selection, DirectML
  packaging, ordinary manual-weight MoMask, batched MotionBricks and Kimodo capabilities. All
  twelve Windows/Linux CTests pass. A packaged Windows three-action smoke reaches
  6/6 targets with one session load; all three GLBs are byte-identical to the
  evaluated baseline. A one-chunk target at `[100,100]` exports ten frames with
  `reached:false` and 141.44 m error, confirming budget exhaustion is reported
  without a synthetic endpoint correction. Linux TGZ installer/reinstaller and
  refusal tests pass; its staged executable generates/reimports the goal example
  without LD_LIBRARY_PATH and with PATH `/usr/bin:/bin`. Native Linux preparation
  from a raw-only cache also matches every graph/tensor pin. The Windows package
  includes the concurrent ARDY DirectML runtime; DART itself uses only CPU ORT.

Foot drift, support-leg/action errors, discontinuities, collisions and uneven
ground remain limits. Imported-history conditioning, arbitrary CLIP prompts,
HML/SMPL-H variants and gradient-based inbetweening are not delivered. Goal
control is a forward-only learned noise policy, distinct from ARDY's separate
native SOMA continuation implementation; it does not need backward graphs.
The earlier gradient/operator assessment remains historical context in
[dart-feasibility.md](dart-feasibility.md).
