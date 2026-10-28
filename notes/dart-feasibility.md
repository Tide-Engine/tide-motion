# DART native backend investigation — 2026-10-28

Status: **not an installed or supported backend**. Checkpoint integration is
blocked on the exact normalization/calibration asset grants below. Forward-only
continuation and policy goal reaching are technically separable from gradient
inbetweening. No DART model inference, conversion parity, generated GLB, goal
accuracy or packaging validation is claimed. Existing backends/defaults are intact.

## Evidence and scope of the rights decision

Source: [zkf1997/DART](https://github.com/zkf1997/DART/tree/bb67ae1ed6ce051080468bf15bc6e54a6c3f8417),
revision `bb67ae1ed6ce051080468bf15bc6e54a6c3f8417`. The author explicitly confirms
Apache-2.0 for **code and checkpoints** in
[issue 25](https://github.com/zkf1997/DART/issues/25#issuecomment-3806649923),
while retaining external project/dataset terms. This is positive primary evidence
for the author's checkpoint distribution/inference grant, beyond a code-license
badge. It is not an explicit grant for every file under the release's `data/`.
The public issue list and this discussion were inspected; no one was contacted.
Upstream source license is preserved in `licenses/DART-SOURCE.txt`.

The [release folder](https://drive.google.com/drive/folders/1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps)
was inspected through public directory metadata. Only four small configuration
files were fetched; no released weights, normalization pickle, body model or
stand/evaluation motion was downloaded for inference. Directory IDs identify
mutable candidates, not content pins suitable for installation.

Candidate artifacts (paths relative to the release folder):

| Artifact | Public Drive file ID | Evidence / unresolved scope |
| --- | --- | --- |
| `mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt` | `1YqWbBEoMCXLvt3CT4yJMFxtuFSCiEnpE` | Author's Apache checkpoint statement; continuation/inbetween demos |
| `mld_denoiser/mld_fps_clip_euler/checkpoint_300000.pt` | `1eIKqohAcIkQHtu3W5IirMiMCfm9Iv4vB` | Different model selected by goal policy args; do not interchange |
| `mvae/mvae_fps_clip/checkpoint_200000.pt` | `1s_KsP7CqCd5s8Y2NQ4qQv6xw_jeAEFbx` | Shared decoder checkpoint, author's Apache statement |
| `policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/iter_2000.pth` | `1q9Vahpx8Ix9QtyoR55ZHPmAld1oFUH52` | Separate trained policy selected by demo, author's checkpoint statement |
| `data/seq_data_zero_male/mean_std_h2_f8.pkl` | `1pz8H_0U0daO_-bvvzaG--zNkqDG91dIl` | Required 276-feature statistics; exact commercial inference/redistribution grant not identified |
| `data/stand.pkl` | `1Qy_zj9FIBX1qwqRUqDQlilqQ4cGDIU66` | Upstream seed motion; provenance not established; exclude and replace with owner-authored history after encoding parity |

For any ID, the metadata URL is `https://drive.google.com/file/d/ID/view`.
No weight or statistics SHA-256/size is asserted: their bytes were deliberately
not fetched after the asset-rights gate. No install manifest accepts these IDs.

Separate legal questions must remain separate:

* [AMASS terms](https://amass.is.tue.mpg.de/license.html) restrict dataset access,
  redistribution and training for commercial use. Tide would not need that
  dataset at inference, and would neither access it nor train on it. These terms
  alone do **not** establish that an inference-only recipient of an Apache
  checkpoint is prohibited. Conversely, the author cannot grant rights held by
  others; the public statements do not explain whether the exact statistics or
  any other bundled derived runtime assets are included in the checkpoint grant.
* [SMPL-X model terms](https://smpl-x.is.tue.mpg.de/modellicense.html) separately
  restrict the full parametric model/software and identify a commercial licensing
  route. DART's stock inference imports construct male/female SMPL-X layers even
  for headless use. Disabling mesh export alone does not remove that dependency.
* [SMPL-X Body terms](https://smpl-x.is.tue.mpg.de/bodylicense.html) explicitly
  permit a CC BY 4.0 subset including a skeleton rig, excluding shape blendshapes
  and tools that create bodies using them. This is a plausible permissive route,
  not a license for the full NPZ model. No exact compatible zero-beta male
  locked-head skeleton/calibration artifact carrying that grant was found in
  the inspected release. The official
  [Unreal plugin](https://github.com/PerceivingSystems/smpl-unreal) also explicitly
  separates its MIT code from body assets; it is not an asset-license workaround.
  The SMPL-X download page redirects to sign-in; no login or acceptance occurred.

**Exact missing external evidence:** a commercial inference/conversion grant for
`mean_std_h2_f8.pkl`, plus a redistributable, versioned skeleton-only reference
for DART's zero-beta male SMPL-X convention (rest joint positions/axes and the
three-component pelvis calibration offset), with an explicit compatible grant,
or an explicit grant to derive/use these from the required full SMPL-X model.
The existing Body license is promising, but no matching licensed artifact and
provenance chain is established. This is an unresolved asset permission and
availability gate, not a finding that all DART weights are research-only.
No public statement reviewed resolves these exact files. Under the owner's
all-components rule, do not quietly classify the statistics as model weights,
extract restricted NPZ values, copy a third-party template, or invent an offset.

## Can inference avoid a user SMPL mesh?

Yes in principle, for a **specific** forward path, with further parity work:

* `PrimitiveUtility.get_blended_feature(use_predicted_joints=True)` uses the
  decoder's 22 predicted joints to construct the new hip heading/origin. It
  rotates all redundant features directly and does not call mesh/FK inference
  each chunk. The misleading method name is not a requirement to blend poses.
* However, reset calls `calc_calibrate_offset`; canonicalization and world
  reconstruction use this offset, even with predicted joints. It is computed
  by invoking the body model at the requested shape. At fixed male/zero-beta it
  could be a licensed constant, but no exact permissively supplied constant was
  identified. The translation formula is `R*(transl + delta) - delta + T`.
  A zero offset changes history inputs and cannot be presented as parity.
* The initial two frames need exact normalized redundant features. Owner-owned
  history avoids distributing `stand.pkl`, but encoding it with an arbitrary
  SOMA skeleton does not reproduce SMPL geometry. A licensed fixed reference
  and independently tested reverse mapping are required. Decoder-predicted
  `joint[0]-transl` is an approximate network output, not evidence for the exact
  rest calibration. Substituting it is an unvalidated model change.
* The default composition demo instead uses `use_predicted_joints=0`, recomputing
  joints from rotations with the body model. That path needs more than the
  predicted-joint route. Goal policy configuration explicitly uses predicted
  joints, making it the best bounded first candidate after clearance.

Thus a full parametric mesh should not be a shipped user dependency for the
proposed bounded path, but its exact replacement has not been cleared or tested.
No claim that installing SMPL-X is unavoidable for all possible native ports.

## Actual source contracts and native work required

The four downloaded release `args.yaml` files are YAML strings containing tyro
tagged YAML. They were read as text, not instantiated/deserialized. The repeat
denoiser selects a transformer with width 512, FF width 1024, eight layers, four
heads, `[B,1,256]` noise and `[B,2,276]` history. Its attached VAE selects
`all_encoder`, width 256, FF width 1024, seven skip-transformer layers, four
heads, `[1,B,256]` latent and eight future frames. Class defaults of latent 128
and VAE width 512 are **not** the released configuration. The distinct euler
checkpoint's args confirm these architecture dimensions and shared VAE; its
weights remain a separate artifact, not an alias of the repeat checkpoint.

`utils/smpl_utils.py` feature layout is:

| Offset | Count | Value |
| --- | --- | --- |
| 0 | 3 | translation |
| 3 | 132 | root/global orientation plus 21 parent-local rotations, row-6D |
| 135 | 3 | translation delta |
| 138 | 6 | global orientation delta, row-6D |
| 144 | 66 | 22 joint positions |
| 210 | 66 | 22 joint deltas |

This is 276, not HumanML's commonly used 263, Kimodo's 369 or MotionBricks' 366.
PyTorch3D's 6D convention stores the first two **rows**. MotionBricks' column
decoder cannot be reused unchanged. Normalization is `(x-mean)/std`, using the
external `[1,1,276]` statistics. Latent scaling multiplies by `latent_std` on
decode; the loader has compatibility buffers for mean/std but decode does not
subtract a latent mean. Export actual buffers and exact rescale flag.

Text uses OpenAI CLIP ViT-B/32, 77-token context, truncation enabled, raw 512-D
text embeddings (no extra L2 normalization). Empty text is explicitly zeroed;
compound actions are sorted and joined with ` and `. Pin tokenizer vocabulary,
BPE merges and Unicode/tokenization behavior as well as the text tower.
Candidate official source pin is OpenAI/CLIP
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (MIT);
its `clip/clip.py` specifies
`https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`.
That path contains upstream's expected SHA-256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`;
not downloaded/independently verified here. Its official model card's research
deployment recommendations must not be confused with an NC license. No CLIP
component is bundled by this change, and its full notice audit remains a
prerequisite to a future distribution. Existing Kimodo text embeddings are not
CLIP embeddings and cannot simply be substituted.

The source sampler predicts **x0**, not epsilon. `create_gaussian_diffusion`
uses a cosine beta schedule, ten steps in release args, fixed variance and
unscaled timesteps. Preserve spaced-timestep mapping, variance, final-step noise
masking and separate DDPM versus eta-zero DDIM equations. CFG evaluates both
text and zero-text branches, returning `u + scale*(c-u)`, with default scale 5
in the demos. Fixed noise tensors, not just equal integer seeds across RNG
implementations, are necessary for parity.

Host rollout emits the two history frames once, then appends eight frames per
primitive: `2+8*N` at 30 Hz (HML variant is 20 Hz and SMPL-H, a separate asset
audit). Carry the last two predicted frames, denormalize, reframe using first
history pelvis and hip line, rotate positions/deltas and root orientation,
conjugate orientation deltas, update world transform and normalize again. Do not
reset the world origin on every prompt. One primitive is 0.2667 s; prompt
segments are integral primitive counts in upstream demos.

Native ONNX inference candidates need three graphs (text tower, denoiser,
decoder); policy adds an actor graph. The VAE encoder is not required for plain
rollout; it is used for training/latent encoding. Expected graph operators are
MatMul/Gemm, Add/Mul/Div, Reshape/Transpose/Concat/Slice/Gather, Softmax,
LayerNormalization and GELU; text also needs causal attention, embedding gather,
EOT selection and QuickGELU (Sigmoid/Mul). Policy uses linear layers,
LeakyReLU/Tanh/Exp/Clamp and normal sampling in the host. Export eval graphs at
fixed H=2/F=8 with explicit tensor arguments, decompose attention, disable
training/dropout and avoid Python dictionaries/distributions in graphs. This
operator list is source analysis, **not an exported graph inventory**; exporter
support, all attention paths, FP16/FP32 parity and actual memory remain untested.
No inference-only `Ort::Session` API supplies arbitrary input VJPs.

Goal policy: `control/test_reach_location_mld.py` loads its own `args.yaml` and
actor weights, then executes without test-time backprop. Observation order is
goal direction (3), distance (1), CLIP (512), history (552), scene scalar (1):
1069 values. Goal heading is clipped to 60 degrees and distance to 5 m. The
policy samples a Gaussian action (256 values here), used as the DDIM initial
noise, with tanh-scaled mean where configured. Using only the mean would be a
behavior change. Success is tested on the last-frame pelvis at a 0.3 m threshold;
goals/text advance on success, not a time-only schedule. Learned coverage in
the released policy config is walk, run and hop on left leg. Do not promise
arbitrary action-conditioned target control. The scene observation is a scalar;
this policy is not the separate SDF scene-interaction optimizer.

Inbetweening/trajectory/scene optimization: `mld/optim_mld.py` uses Adam over
per-primitive initial noise and `ddim_sample_loop_full_chain`, differentiating
through all denoising steps, decoder and history/world transforms. Goal loss is
Huber over selected final joints, with optional floor/jerk penalties; other
scripts add trajectory/SDF terms. The demo uses 100 optimization steps, ten
DDIM steps, unit gradient normalization and learning-rate annealing. A 15-chunk
case entails 30,000 denoiser branch evaluations and 1,500 decoder evaluations
before the final rollout, plus backward work. Ordinary forward continuation at
the same length needs 300 branch evaluations and 15 decodes. These are static
call counts, not speed measurements. Noise alone is 15 KiB at B=1/N=15/256/F32;
Adam state and activation tapes dominate beyond that, and peak memory has not
been measured. Options for native gradients are separately exported VJP graphs,
a tested native backward implementation, or an explicitly pinned training
runtime; finite differences through thousands of noise variables is not a
credible substitute. Gradient work does not gate the forward-only policy path.

## SOMA conversion and validation gates

DART's world is Z-up; use a proper basis, e.g. `(x,y,z)->(x,z,-y)`, conjugating
rotation matrices and separately calibrating heading. Decoder root rotation is
global; 21 body rotations are parent-local. Network joint predictions are
redundant features and can disagree with rotational FK. Export must use fixed
canonical SOMA bone offsets and unit local rotations, never animate bone lengths
to fit predicted joints. Use a pinned explicit SMPL-to-SOMA mapping/rest-axis
calibration. Missing SOMA twist/end joints need a documented mapping policy,
not automatic identification. Root translation must include the calibration
and world transforms before scale/heading conversion. HML 20-to-30 Hz conversion
needs shortest-arc quaternion interpolation and duration/end-frame checks.

Not implemented/validated: the actual DART-to-SOMA mapper, reverse history
encoder, independent FK comparison, resampling tests, real end-to-end inference,
CLI/batch/install/recipe integration or package tests. These are technical work
pending the external asset gate, not technical impossibility findings.

Once exact assets are cleared, first export/pin only the forward policy path,
compare fixed tensors independently at text/denoiser/decoder/reframe boundaries,
then implement bounded host continuation/waypoints, one session set and text
cache per RunContext. Preflight segment/target budgets and asset rights/hashes
before loading. Recipes must record every checkpoint/conversion hash, sampler,
noise seed, actual device, primitive schedule, targets and measured errors.

Use owner-generated idle/walk history and own prompts, eight fixed seeds with
equal noise tensors: straight target, left/right corner, changing walk/run goals,
and a long sequence with bounded primitive/timeout limits. Measure last-frame
and minimum goal error, success fraction, seam root distance/velocity and joint
angular steps, action adherence from an explicit rubric, and foot drift using
FK feet/contact thresholds on SOMA. Record raw uncorrected results, initialization
versus warm timing and peak process/device memory. Do not borrow upstream >300
fps as a Tide measurement. Validate saved GLBs with the existing strict loader,
fixed bone lengths, finite values and independent FK, then inspect bounded
`TIDE_HIDDEN=1` captures using Tide built from the validated engine checkout.
Take `TideMotionGpuValidation` for any expensive GPU reference/native runs.

The distinct value beyond the separately implemented ARDY SOMA continuation path
is DART's released noise-policy waypoint controller and eventual optimized
inbetweening, with their own target-success and gradient tests. Multi-prompt
continuation alone overlaps ARDY and does not justify claiming the goal features.
No ARDY implementation/shared integration files are changed here.

## Source-only proofs actually run

`tests/dart-feasibility` is an isolated C++23 CMake project; it adds no production
backend, model download, runtime dependency or installed artifact. Its mathematical
fixtures are independently authored and contain no motion data or trained weights.
On Windows x64 ClangCL 18.1.8 Release:

* Row-6D decoding of an independently specified 90-degree rotation, invalid
  rows, pelvis-offset/world inverse, and proper Z-up/Y-up handedness pass.
* Six eta-zero x0-DDIM local adjoints agree with central finite differences to
  maximum absolute error `1.67395e-11`. This does not test network/rollout VJPs.
* An optional in-memory, synthetic ONNX opset-20 graph executes
  MatMul/LayerNormalization/Gelu on pinned ORT 1.31.1 CPU, agreeing with an
  independent scalar reference to `2.37287e-08`. The shipped inference DLL returns
  null for `GetTrainingApi(1)`. Training headers being present does not enable
  backprop. No DART ONNX export or attention graph is exercised by this fixture.
* Both isolated CTests pass (0.17 s observed, not a benchmark). No GPU used;
  inference time, model memory, goal error, seams, adherence and foot drift are
  **unmeasured**, since no DART generated asset exists.

Reproduce (use a unique build directory per thread):

```powershell
$env:TIDE_HIDDEN='1'
cmake -S tests/dart-feasibility -B build-dart-probe -G "Visual Studio 17 2022" -A x64 -T ClangCL -DMOTION_ORT_ROOT=PATH_TO_PINNED_ORT_1_31_1
cmake --build build-dart-probe --config Release
ctest --test-dir build-dart-probe -C Release --output-on-failure
```

Omit `MOTION_ORT_ROOT` for the dependency-free math probe. The optional ORT fixture
is Windows-only. No new full-product test run is warranted by these isolated
source-only additions; production code/build/install logic is unchanged.

## Content pins

SHA-256 of LF UTF-8 source bytes at the source revision above (URL base:
`https://raw.githubusercontent.com/zkf1997/DART/bb67ae1ed6ce051080468bf15bc6e54a6c3f8417/`):

```text
LICENSE b505b57d3d1a8d897cdcc74cfb0807f72405c2686100568475fa8f1184594a84
README.md 4a15a495b1d3c13264bafeb4c1e1b747bbbdad697e326396d32eb40dca68568d
model/mld_vae.py 1c4fbf6b749f22ef1fa1f489412c4d3fc58cd3d4f772e2fa17be5dbb4050534b
model/mld_denoiser.py 9e14e238007642299081446b427446966a44259df64a3bd15c898e2ffd639b89
utils/smpl_utils.py ec20124b489cf9f084fd13186190ba9376139f3366e3d33e4194c9906d8899dd
diffusion/gaussian_diffusion.py a813e9b6b1c306ba740a4e62f3e5ec4c29da3f3196be0a18699aad2f71cdf06c
```

Configuration file download URL pattern:
`https://drive.google.com/uc?export=download&id=ID`; hashes are exact fetched bytes:

```text
repeat args.yaml 1_mktoa64Gc8-Mcf3G4u_WeDXxjszj-qE ed47686b2a90995bf24d3cb54176e397477656120a5d94592e001104a06e16c1
euler args.yaml 15jEXTPpKFMBVQ_FJclqW9k1GRrYEFtsL b8f068970284fecbc837814c23041070c7800fe922da805ea299f8800b3da86a
VAE args.yaml 1MfWZnB1_U1wdrMNJoO-bOc44vMUkmcM7 ff59035742622b349f872577e1c232716987397464197cd551a8972816c29f53
policy args.yaml 1thw7mncpduN7B4APGhBnOEGCm970XDLW 25f2b126e5c567540cd1766606151cb79322854b023fc97ba604e39799254e5e
```

ORT is the product's already pinned Microsoft native CPU 1.31.1 Windows x64
archive, SHA-256 `5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c`
at `https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-win-x64-1.31.1.zip`
(MIT notices already in the product). No dependency revisions changed.
