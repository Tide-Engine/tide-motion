# Native MoMask with manually supplied weights

The owner explicitly authorized local development/use with unresolved pretrained
checkpoint license scope after the commercial-rights investigation. The source
code is MIT licensed upstream; the exact pretrained weights/statistics still have
no verified separate author-issued grant covering commercial use. This backend
does not assert commercial clearance. Weights stay private, outside Git/releases.
The earlier primary-source findings remain in `notes/momask.md`.

Current controls also include native Windows DirectML GPU acceleration, variants/
contact sheets, CPU thread controls, in-place generation, SOMA-77 export and
source-extra-track preservation. See [the capability comparison, measured GPU
checks and model limits](momask-capabilities.md). Historical experiment/build
results below remain labelled by their original integration stage.

## Use on this Windows machine

MoMask is included in ordinary Tide Motion builds. Import the supported prepared
ONNX bundle manually, then use the normal executable. No model download is
performed, and existing backends/defaults remain unchanged:

```powershell
tide-motion models import-momask C:\path\to\prepared-bundle
tide-motion models check momask
tide-motion generate "A person walks forward." --backend momask --duration 3 --seed 7 -o walk.glb
tide-motion generate "A person stands still in a relaxed pose." --backend momask --duration 3 --seed 2 -o idle.glb
tide-motion inpaint "A person waves with the right hand." --source idle.glb --begin 18 --end 66 --seed 3 -o wave.glb
```

`--begin` and `--end` select `[begin,end)` on the original **30 Hz** source clip.
The range must span at least 12 frames. Both endpoint poses are retained, with
the learned interior corrected to them. Every source pose outside the range is
copied, with no tail truncation. GLB serialization may normalize/negate quaternion
keys for shortest-sign continuity; the represented outside poses remain unchanged
(measured worst component roundoff 1.20e-7 after matching quaternion hemispheres).
Source GLBs must already have Tide's canonical SOMA-30 or SOMA-77 rig. Use existing explicit
mapped-humanoid retargeting before/after this tool for other humanoids.
SOMA-77 sources preserve all extra-joint tracks at every frame, with body edits
through the canonical SOMA-30 subset. Other custom extra joints are refused.

Outputs are ordinary skeleton-animation GLBs plus `.glb.json` recipes. Existing
output/recipe files are refused. Prompts must be ASCII, without HTML entities or
special-token strings; non-ASCII is refused rather than tokenized incorrectly.
No automatic rig identification, fingers/facial prediction, skinning or
non-humanoid retargeting is introduced. English prompts are examples, not hard
semantic constraints. Feet can slide/penetrate the floor; contact IK remains a
separate existing optional Tide Motion operation, not part of this model.

Controls: duration .2..600 seconds (generation only), seed 0..4294967295,
steps 2..1000 (18 default), guidance 0..10 (4), residual-guidance 0..10
(generation 5 / editing 2), temperature (0,10] (1), topkr [0,1) (.9).
Motion lengths floor to four native **20 Hz** samples/token. Conversion samples
the same interval at 30 Hz without a fabricated tail: requested 3 s yields 60
native samples and 89 exported samples; requested 2 s yields 40 and 59. First
useful prompt examples are 2–4 s. No Kimodo path/effector/keyframe controls are
accepted. CPU is the compatibility default, using four threads unless --threads
selects 1..1024. --device gpu requires native Windows DirectML, auto tries it and
falls back to CPU; CLIP text stays CPU. --variants 1..64/--variants-gif/--tide reuse
the shared variant renderer. --rig soma77 adds static relaxed hands on generation;
source SOMA-77 tracks are preserved on inpaint. --in-place is generation-only,
so it cannot silently change outside an edit. There is no user Python, hosted
service, daemon or procedural/blended motion generator in this inference path.

`--models CACHE` or `TIDE_MOTION_MODELS` selects the normal cache; MoMask lives in
`CACHE/momask`. `models import-momask BUNDLE_DIRECTORY` is a native **local import**,
not an Internet model download. It verifies the complete pinned manifest and every
file, copies to staging, verifies the copy and renames it. Existing destinations/
staging paths are refused; failed staging remains available for inspection.
`models check momask` hashes the bundle without loading inference sessions.
Raw upstream `.tar`/`.pt` files are not a prepared bundle. `models install momask`
explains manual import; `models install all` is unchanged. Model rights are not
changed by import, and no weights/statistics/vocabulary are shipped.

## Ordered batches and recipes

`tide-motion batch JOBS.json` uses the standard `actions` manifest; see
`examples/momask-local.json`. Generation actions select `backend: "momask"`;
editing actions use `action: "inpaint"`. Each has `prompt`, `output`, optional `duration`,
`seed`, `steps`, `guidance`, `residualGuidance`, `temperature`, `topkr`. Editing jobs
add `source`, `begin`, `end`. Paths are manifest-relative. All arguments, existing
sources and output/recipe collisions are checked before loading models. Sources
may also be outputs of earlier actions in the same batch; their clip validation
is deferred until produced. One model context verifies files once and reuses all
five ORT sessions across interleaved actions. A failed job stops and keeps completed
outputs. Device/threads/variants/rig/inPlace fields are supported too. This does
not add learned Kimodo spatial inputs or ARDY autoregressive inputs to MoMask;
current token-context and postprocessing controls are listed in the capability
document.

Recipes record the actual bundle hash, rights status, prompt, seed, sampler/RNG
version, effective controls, CLIP token IDs, base/final RVQ token IDs, source
placement/edit range, uncorrected endpoint root errors, postprocessing, frame
counts and native inference/conversion time. Token IDs are generated data; no
training/evaluation motion is imported. Prompts and recipes stay local.

## Architecture and conversion

Same exact upstream source/configuration revisions as the earlier study. Developer
export loads only hash-pinned weights with Torch `weights_only=True`. Pinned
private inputs are in `notes/momask-private-inputs.json`; exact native runtime
assets are in `notes/momask-private-bundle.json`. Native code also pins the hash
of the whole bundle manifest before checking each file. Runtime assets total
about 473 MB (451 MiB), excluding the original development checkpoints. No weights, ONNX
graphs, normalization data, tokenizer vocabulary or binary fixtures enter Git.

Native ASCII CLIP byte-BPE follows the official token splitting/merges, lowercases
ASCII, truncates to 77 tokens including start/end and pads. An exported official
CLIP ViT-B/32 text tower returns the unnormalized 512-vector. Masked and residual
transformers run conditional/unconditional step graphs through existing ORT
1.31.1. All loops, confidence remasking, CFG, top-k, categorical/Gumbel sampling,
history embeddings and codebook-token orchestration run in C++. The versioned
SplitMix64 uniform stream intentionally differs from PyTorch's seeded RNG;
equal seed numbers across backends do not promise equal motion. Real decoder
outputs are denormalized and converted through native HumanML math to canonical
SOMA before GLB export/retargeting.

Inpainting samples the canonical source at 20 Hz, pads its end to whole tokens
and encodes all 263 features natively. The explicit 22-joint correspondence,
metre units, hip-derived heading, initial world placement and incoming-edge
rotation conventions are tested independently. This avoids dataset-derived
example skeletons, SMPL/SMPL-X assets and any body mesh. The current encoder uses
canonical source proportions and per-frame heading, without upstream's dataset
skeleton normalization/Gaussian heading preprocessing. It is not claimed as
bit-identical upstream feature extraction on arbitrary rigs. Source windows are
bounded at 196 native samples (at most 293 canonical source frames).

Edited base tokens cover the requested range conservatively; residual tokens
are regenerated across the window, as upstream. Decoded cumulative root drift
at the two edit endpoints is distributed linearly across the edited interval;
existing native pose correction then makes the two body endpoints exact over
10-frame neighbourhoods. The original outside poses are restored last. This is
documented postprocessing of real learned inference, not replacement generation.
One walking edit initially had a 1.81 m endpoint root error and 8.49 m/s corrected
peak speed with short-window root correction alone; distributing drift reduced
the peak to 2.04 m/s. Neither correction guarantees velocity/acceleration or
contact continuity. Recipes retain the pre-correction errors.

## Executed validation

Windows x64 ClangCL 18.1.8 Release. Eight CTests pass, including native HumanML
normalization/root/FK/SOMA-encode RIC roundtrips and CLI/batch preflight/refusals.
The new CLI test's initial CMake semicolon-list fixture was corrected; its rerun
passes. Worktree-local TEMP/TMP avoid concurrent test collisions. Project code
builds with warnings as errors. These initial experiment results precede ordinary
CLI integration; current integrated checks are recorded below.

Real native ORT 1.31.1 versus PyTorch: all 13 component cases pass (masked/
residual/RVQ at 4/7/49 tokens plus CLIP at 77 tokens). RVQ indices match exactly;
maximum errors are 8.59e-6 masked logits, 1.63e-5 residual logits, 8.82e-4 RVQ
decoded normalized features and 1.44e-6 CLIP. Bounds are 3e-5 + 3e-4*abs(reference)
for the native checker. Larger decoder absolute error occurs on large-magnitude
random code combinations; no bitwise arithmetic parity is asserted.

`tests/momask-reference.py` independently reproduces full masked/residual token
selection with Python/NumPy/ORT and checks tokenization against the official CLIP
tokenizer. Every base/final token matches for real walk, idle and contraction/
punctuation prompts. Independent exported-track checks on idle->wave and
walk->arms edits verify all outside poses and both endpoints within 1.20e-7 per
component after equivalent quaternion sign matching. Native GLB import/inspect
accepts all outputs. Two-job batches load each model session only once. An
installed native generation succeeds with PATH restricted to Windows system
directories; no Python dependency is present. The existing Tide Motion install
was not replaced during the initial separate-launcher validation.

CPU observation: two 2-second jobs with shared sessions complete in 1.34 seconds
including process startup, bundle hashing and loading; inference/conversion
0.125/0.133 seconds. Coarsely sampled peak working set 552,628,224 bytes (50 ms
sampling). This is a local observation under concurrent work, not an isolated
benchmark or GPU speed comparison. No GPU work ran, so no GPU mutex was needed.

Quality comparison uses the same two own-authored prompts, requested duration 2 s
and seeds 2 (walk)/1 (idle), against cached native Kimodo v1.1 50-step outputs.
MoMask uses 18 mask steps and exports 59 rather than 60 samples. Independent FK
proxies, not upstream evaluation metrics: walking root travel 2.458 m (MoMask)
and 2.042 m (Kimodo); idle travel .015/.013 m. Mean horizontal foot/toe movement
for samples below 3 cm, cm/frame: walking 2.085/.312, idle .282/.027. Minimum
foot/toe height: walking -.022/.011 m, idle -.027/.017 m. The different skeleton
reconstruction and negative foot heights affect which samples enter the proxy;
these four clips do not establish general backend ranking. Contacts currently
look worse in these MoMask samples; no superiority or artist-ready quality claim.
Kimodo assets are previously self-generated, not evaluation data. No Kimodo
inference was repeated for this comparison and no new Kimodo speed measurement
is asserted.

Hidden bounded mannequin captures: 61 frames for generated walk, 85 frames for
edited idle wave, 960x540. Used native Tide rebuilt from D:/Projects/PurrEngine
at `28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c`, with its matching MinGW platform
staged privately. The broad engine installer stopped at an unrelated unbuilt
editor after staging CLI/engine files; platform-only staging completed. No engine
source changed. Visually inspected coherent walking limbs and a raised right
hand without inverted joints or binding errors; sparse captures do not establish
temporal/contact quality. TIDE_HIDDEN=1 was used for validation and previews.

## Build/reproduce locally

The backend is built normally; selecting it and supplying weights are explicit:

```powershell
cmake -S . -B build-local
cmake --build build-local --config Release
ctest --test-dir build-local -C Release --output-on-failure
cmake --install build-local --config Release --component runtime --prefix YOUR_DIRECTORY
```

`MOTION_MOMASK_EXPERIMENTAL=ON` now only adds the legacy standalone launcher;
ordinary `tide-motion` includes MoMask even when that flag is OFF. The compatibility
`tide-motion momask ...` namespace remains, but the commands above are preferred.
No existing backend/default is replaced.
Developer-only preparation: `tests/momask-download.ps1` verifies private input
pins; `tests/momask-export.py SOURCE OUTPUT CHECKPOINT_DIRECTORY` exports and
validates the private bundle. Its locked Python environment is not shipped or
used by inference. `momask-ort-tests BUNDLE` checks all saved reference fixtures;
`tests/momask-reference.py SOURCE BUNDLE NATIVE.glb [EDIT.glb BEGIN END]` checks
sampling/tokenization and source preservation. ONNX/reference artifacts stay in
ignored scratch. Preserve notices; do not upload/private-copy the weights into
Git or release archives. Checkpoint commercial scope remains unresolved.

Remaining limits: Windows CPU/DirectML validated, ASCII tokenizer scope, one prompt
per job, published-window bound, lossy SOMA rotation reconstruction/proportions,
no predicted fingers/facial motion, contact/velocity continuity and broad quality
coverage. Commercial distribution remains a separate unresolved decision.

Integration: rebased onto origin/dev 706665e, preserving native ARDY, measured
Kimodo selection/action guidance, cleanup and MotionBricks joint-frame/guide/
contact changes. The integrated ClangCL Release build passes all **nine** CTests.
The cosine schedule now follows float32 operation order: FP64 changes remask
counts at half-integer boundaries (3 tokens / 4 steps). An independent real
short-clip run verifies that boundary and full token parity; the earlier default
18-step walk token sequence remains unchanged. Windows case-insensitive/canonical
output collision, whitespace prompt and existing partial-file refusals are also
covered. Rebase changed no model graph/bundle pin or numerical tolerance.

Final integration includes origin/dev 68dd1c5's native upgrade and batched
MotionBricks work too. All **ten** integrated CTests pass on that rebuilt tree.
The separately installed owner launcher is refreshed from this checkout; no
commercial release/model distribution or ordinary-install replacement was made
in that earlier step.

## Ordinary CLI/manual-import integration (2026-10-28)

The normal build with the legacy launcher flag OFF passes all ten CTests. Native
import/check verifies a real 473 MB prepared bundle. A real four-action standard
batch (walk generation, mirror, idle generation, inpaint consuming that idle)
loads the five sessions once. All outputs are valid 59-frame SOMA GLBs. The walk
animation payload is identical to the previous native experiment. Independent
reference sampling/tokenization still matches; exported outside/edit endpoint
components differ by at most 1.1920929e-7 after quaternion hemisphere matching.
No model graph, sampler, numerical tolerance or asset pin changed. Default builds
carry only source/native binaries and notices, never the separately supplied assets.

After rebasing onto bb25953, the full Vulkan/ARDY DirectML build passes all eleven
CTests. The ordinary runtime ZIP passes install/self-upgrade checks and includes
asset-pin documents, without any assets. The owner's normal Tide Motion install
is updated, preserving the four existing model files and PATH; the prepared
MoMask bundle was imported separately. Running the installed standard example
with Windows-only PATH produces three 89-frame SOMA clips with one model load.
Independent token/reference and all outside/edit endpoint checks pass, maximum
component error 6.5192581e-8 after quaternion sign matching. At that stage extra-joint
sources were refused; the later capability work adds explicit SOMA-77 extra-track
preservation, independently tested on an authored varying finger track. No new
predicted fingers are claimed.
