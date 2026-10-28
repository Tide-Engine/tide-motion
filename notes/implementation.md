# Native animation authoring

## Native ARDY backend (2026-10-28, supersedes the initial spike limit)

The owner authorized complete implementation and developer-only reference/export
tooling. The earlier separate-budget and 2x-value gates were not owner requirements
and are withdrawn. See [ardy-native.md](ardy-native.md) for current setup, pinned
artifacts, actual bundled commercial grant, contracts, reproducible tests and limits.

Integrated five-action native CPU batch: cold and warm 183-frame walk, 123-frame
imported-history walk-to-wave with a path, 183-frame full-body/hand/path-constrained
walk, and 1,800-frame (60 s) walk. Requested counts, finite FK, GLB save/load and
same-seed warm repeatability pass. One ARDY session set and one text verification
are logged for the whole batch. Wall time 63.6008 s, peak working set 9,188,741,120
bytes. Per-action generation (includes text, excludes session loading): 22.1753 s
cold, 6.44175 s warm, 5.44895 s history/prompt change, 6.54914 s constraints,
20.3531 s long-form (4 steps; others 10). The runner held TideMotionGpuValidation,
but a separate t3-3e98d7db arms-selection process ran outside it: timings are
contended observations, not an isolated comparison or speedup claim.

Uncorrected constrained clip: full-body position mean/max 1.57/4.01 cm, right-hand
position 1.45 cm, path hips-ground error 3.27 cm. Bounded captures at 221/181/1801
frames of history-to-wave, constrained walk and long walk succeed; front/profile
images were inspected. Walk-to-wave visibly raises the right hand while walking.
No gross limb inversion or binding error was seen. The strict 3 cm heel-height
slide diagnostic had zero qualifying samples, so it reports unavailable, not
zero sliding. Continuous contact quality is not certified. Integrated CTests
include concurrent cleanup/HumanML work: all eight pass.

Final runtime-only Windows CPU archive: 8,281,696 bytes (no weights or developer
Python tooling). The final isolated installer test passes again. The packaged
executable performs real ARDY inference and saves six requested frames with PATH
restricted to Windows system directories; neither Python nor development tools
are available to that child process. The same external pinned model bundle is
used, and all three native sessions plus local GGUF text inference load normally.

`generate --backend ardy` and matching batch fields run complete denoiser/tokenizer
encoder/decoder graphs through existing native ORT 1.31.1 CPU, with the pinned
native local LLM2Vec GGUF encoder. C++ owns DDIM, history/masks, origin/requantization,
prompt switching and canonical SOMA GLB export. Imported history and 60-frame
autoregressive chunks are supported. Exact requested output counts include partial
last chunks. The standalone delivered binary never invokes Python. Conversion
scripts and fully pinned developer requirements stay outside runtime packaging.

Pinned source e706e27acc88070918308a10922b8f7174d0825e; commercial SOMA KimodoRep
checkpoint e6c6804fb6b55636ad927145d74959fd46882ccd. Every original checkpoint file
and converted graph is hash checked. Native code pins the conversion manifest
aac3b966ed1368c4ec126c969f0763d59126b047a1053a2394c20983acb917d2. The bundled
October 2025 license grants commercial use, derivative creation and distribution;
no concrete missing right for conversion/native use was found. It and required
Notice attribution are retained and checked on import. The model-card link to a
different commercial agreement is recorded, not used as a technical stop.

Windows x64 ClangCL 18.1.8 Release with warnings as errors passes six CTests.
Actual native ORT network fixtures cover history 0/4/240 plus constraints: max
denoiser absolute error 2.08e-5; decoder 1.54e-3 (mean 1.09e-5); FSQ encoder fixtures
exact. Four-chunk fixed-noise host/path test: 183 frames, max root difference
0.056 mm vs upstream. Imported-history fixed-input encoder/host: 121 frames,
0.26 mm root difference. Native FK/smoother feature error 1.79e-5; including those
operations and FSQ boundary sensitivity, observed output difference is 2.2 mm /
1.84 degrees. Tests distinguish fixed-input parity from that end-to-end rounding
effect. No full FP32 text-encoder parity or bitwise platform determinism is claimed.

Own-prompt native generation exported a 183-frame walk. A bounded 91-frame hidden
mannequin capture passed and was inspected in front/profile using Tide CLI and
tide_platform rebuilt from D:/Projects/PurrEngine at
28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c, privately staged, with the GPU mutex held.
Runtime Windows CPU archive was 8,156,843 bytes before the final documentation
update; isolated install/reinstall/path traversal/checksum/PATH-minimal tests pass.
Native model import/check of the converted external bundle passes. No engine,
Kimodo model algorithm, contact cleanup or MotionBricks authoring changes are made.

Limits: CPU-only ARDY validation on Windows; no ARDY GPU or other-platform claim.
Prompts switch at 60-frame boundaries; future constraints are windowed; imported
history need not produce an exact pose/velocity seam. Contacts, scene awareness and
constraint satisfaction are not guaranteed. Separate backend text instances can
increase batch memory. All weights, fixtures and generated clips remain external
or in ignored scratch; no research-only checkpoint or evaluation motion data is used.

## ARDY feasibility spike (2026-10-28)

Baseline reviewed: v0.5.0 / dd4dc0b. The focused assessment and reproducible native
probe are in [ardy-feasibility.md](ardy-feasibility.md) and notes/spikes/ardy/.
Upstream source is pinned to nv-tlabs/ardy
e706e27acc88070918308a10922b8f7174d0825e; exact commercial SOMA RP checkpoint
revisions, source links, architecture/operator inventory, representation, local
text reuse, chunk/history/constraint contracts, GLB adaptation, runtime/license
risks and staged acceptance gates are recorded there. Both SOMA model cards permit
commercial use, but their linked 2026 agreement differs from the bundled October
2025 LICENSE; preserve the actual bundled terms and resolve before distribution.

The optional standalone C++ probe uses the existing pinned JSON dependency and
no Python, service, simulator or GPU. On Windows x64 ClangCL 18.1.8 Release with
TIDE_HIDDEN=1 and warnings as errors, it validates the hash-verified KimodoRep
denoiser container (627,156,264 bytes, 224 F32 entries, 156,781,716 elements) and
executes its learned 148-to-1024 global-root hybrid affine projection with
synthetic input. Output sum 0.715780863888; max absolute error versus independent
double accumulation 4.42014425062e-7. Oversized-header, truncated-checkpoint and
missing-argument refusals pass. Full SHA-256 and reproduction are in the assessment.
This is native checkpoint/operator feasibility evidence only, not model inference
or PyTorch parity. The probe is outside the CLI/build/install/release targets.

Historical initial-spike decision (superseded by the native backend above): complete denoiser/tokenizer/text parity, imported-history
encoding, DDIM/state-loop execution, canonical rig/GLB validation, long-sequence
quality, memory and CPU/GPU performance remain open. No ARDY generation, preview
or GPU timing is claimed; no GPU mutex was needed for this CPU-only probe. Future
GPU runs must hold TideMotionGpuValidation. Weights and upstream source remain
in ignored scratch; no research-only checkpoint or evaluation motion data was
used or committed. Kimodo v1.2, contact cleanup and MotionBricks authoring are
untouched. Existing runtime behavior and dependencies are unchanged.

## MoMask investigation and native source work (2026-10-28)

See [momask.md](momask.md) for checkpoint-specific rights evidence, exact release
artifact inventory, native conversion/edit contracts and executed operator proof.
Commercial eligibility of the official RVQ/masked/residual checkpoints and their
statistics could not be established from primary release artifacts. The MIT code
grant and AMASS's separate dataset/training restriction do not independently
decide those checkpoint/output rights. A bounded ZIP-directory/configuration
inspection found no checkpoint license text. No checkpoint or evaluation motion
was integrated or distributed; no MoMask backend/default/install entry is exposed.

Lawful source-only work is implemented: C++ HumanML normalization, root/RIC/edge
FK, lossy canonical SOMA conversion/resampling and exact outside-range splice
with endpoint refusal. These compile only into model-free tests. Random-weight
RVQ encoder/decoder and both transformer steps export at opset 17 after fixing
dynamic attention and tied projection export issues; all 12 cases pass native
ONNX Runtime 1.31.1 against PyTorch (encoder IDs exact, worst float error 1.20e-6).
Pinned source/configuration hashes, tooling lock, notices and numerical results
are committed; weights/ONNX/fixtures remain ignored. SMPL/SMPL-X research body
assets can be omitted entirely for skeleton-only inference. Real checkpoint
parity, native CLIP/RNG, source encoding/calibration and end-to-end generation/
inpainting/quality comparisons remain untested, not claimed as functional support.

Integration validation: rebased onto origin/dev 15c78fd, preserving concurrent
ARDY notes, DART evidence, SOMA-RP v1.2, pose authoring/placement and contact
cleanup. Resolved additive CMake/notes conflicts; fresh local Windows CPU Release
build passes all seven CTests with worktree-local TEMP/TMP and TIDE_HIDDEN=1.
Native synthetic canonical GLB export/reload and all 12 native ORT fixture cases
pass; runtime staging includes the report, pin files and notice. Source verification
checks 52 hashes and refuses a deliberately modified source file. No GPU, inference
benchmark, engine preview, PR or manual CI dispatch was used for this work.

Owner-requested Hugging Face recheck: audited all six current name-matching model
repositories' public card/license artifacts plus the unchanged author-linked
Space. Revisions, text URLs and hashes are in notes/momask-huggingface-audit.json;
detailed findings are in momask.md. The compatible geedog mirror carries the
original MIT software text but supplies no original-author checkpoint-scope or
uploader-authority evidence; the complete ZeyuLing conversion says license other.
The MIT-labelled Signvrse publisher model is a 668-feature RVQ tokenizer without
a compatible text generator, not a replacement for the 263-feature bundle.
Commercial checkpoint eligibility remains unestablished; this does not assert
categorical noncommercial status. An applicable standard author checkpoint grant
can clear this gate; bespoke output-license wording is not inherently required.
No weights were fetched, runtime code changed or prior inference tests repeated.

## Scope agreed with the owner

Separate repository, CLI and installer. Native executables with minimal user
dependencies; no required Python environment. Tide's CLI and engine remain
independent. Kimodo creates motions; MotionBricks creates clip-to-clip transitions.
Ordinary exported animation assets are the game interface. Tide can render previews.

## Native contracts

The core is C++23. Kimodo is compiled as a static library from localai-org/kimodo.cpp
at ed8a931aedb132f768190f6931c365855caba74c, including its GGML submodule at
8c63e70982c95ceb862e3a1073a2c1beef75d60a. The upstream port targets the SOMA RP
v1.1 checkpoint. Tide also validates opt-in smooth-root v1.2 using the same
native architecture (see the v1.2 section below).
Native output is the compact 30-joint rig, not the 77-joint presentation skeleton.

MotionBricks uses Microsoft's native ONNX Runtime CPU distribution 1.31.1. Pose
encoding, world alignment, 6D rotation decoding, boundary trimming and endpoint
correction are adapted from NVlabs/motionbricks at
54fc563aeaaf819a753d72cc9fdc6713989cdd43, model/soma/kernels.py and
utils/pose_utils.py. The original copyright and Apache license are retained.
This avoids depending on the unrelated G1-only native MotionBricks port.

Context: last four source samples; target: first four target samples. Both are
30 Hz parent-local poses on the canonical SOMA rig. FK gives positions and global
rotations. Alignment removes the last context root's ground position and turns
its hip-derived heading to +X. Inputs have the official [1,8,5], [1,8,4],
[1,8,267], three [1,8] masks and [1,11] duration flags. Context/target derivatives
at slots 3 and 7 are masked out. Default token counts are 8..14.

Output is [1,64,366] and one token count. Three frames are trimmed at each end.
Global rotations are reconstructed by orthonormalizing the first two matrix
columns. Child translations come from the rig; global-to-local rotations use
the canonical parent indices. Root alignment is restored. Endpoint corrections
follow the upstream local transform and world root correction rules, then both
endpoint poses are copied exactly to avoid numerical residue. No generated
physics or network simulation state is introduced.

The CLI currently accepts SOMA self-contained GLB only. A canonical name lookup
reorders nodes; duplicate/missing names, changed parents/proportions, nonunit
scale, matrix nodes, external buffers, nonfinite values and unsupported channels
are rejected. LINEAR and STEP sampling is at 30 Hz. File size is capped at
512 MiB, skeleton nodes at 1,024, key counts at one million per sampler and output
at 18,000 frames. GLB export writes LE F32, complete local translations/quaternions
and shortest-sign quaternion continuity. Model inference itself is offline visual
content work and does not promise bitwise cross-platform determinism.

Model downloads use HTTPS-only curl argv without shell evaluation, pinned URLs,
sizes and SHA-256. Partial files remain resumable; final files appear only after
verification. The explicit models install command accepts model licenses. Each
generation verifies its selected models. Prompts and recipes remain local.
Windows SHA-256 uses BCrypt (an OS dependency) so verification can use hardware
acceleration; other hosts use the pinned PicoSHA2 fallback. Three resumable curl
invocations also cover dropped TLS connections that curl's own HTTP retry does
not retry. A real 8 GB transfer was interrupted twice and completed from the
existing partial bytes, then matched its pinned digest.

The owner additionally requested batch actions to avoid loading models N times.
`batch jobs.json` validates a bounded 1 MiB / 1,000-action manifest and all output
paths before inference, then executes in declared order with one owned RunContext.
The Kimodo model and ONNX session are each loaded lazily and reused across the
entire run, including interleaved backend actions. Checksum verification occurs
only at the first backend load. Text layers are kept resident by setting the
upstream layer chunk to 32 before loading, unless the operator explicitly set
KIMODO_TEXT_LAYER_CHUNK. No cross-process daemon or persistent server exists.
Failure stops the sequence and keeps completed output assets. Manifest paths are
relative to the manifest, allowing later actions to refer to earlier outputs.

## Validation and remaining work

Windows x64 Release succeeds with Visual Studio ClangCL 18.1.8. All three CTests
pass: math/GLB/model-contract checks, CLI help, and CLI asset/argument/batch
preflight/hidden-preview checks. They cover non-grid frame-time roundtrips,
placement-invariant encoding, derivative/duration masks, boundary trimming,
exact endpoints, seam frame counts, nonfinite and changed-proportion/animated
bone-length rejection, paths with spaces/quotes, malformed GLBs, fractional
integer arguments, refusing existing output, complete manifest preflight before
loading, and license gating. The final Windows BCrypt path verifies every real
model against its pinned digest.

Real native inference: an ordered batch generated two 60-frame SOMA clips
(relaxed idle and walk), then made two 26-frame MotionBricks transitions with
8 tokens and exported joined 144-frame sequences. The logs contain one Kimodo
load and one MotionBricks load for the entire four-action process. Kimodo used
50 diffusion steps, seeds 1/2 and 8 CPU threads. Observed resident memory was
about 9.4 GB while generating. Another two-action MotionBricks batch consumed
the first action's output in its second action with one backend load. The
reproducible opt-in test is tests/inference.ps1; models are installed separately.
These are functional integration tests, not upstream numerical-parity tests or
a claim that every generated motion is artist-ready.

The generated GLBs load through Tide built from the engine checkout at
C:/Users/blenm/.t3/worktrees/PurrEngine/t3-dea14cb9. The native CLI and MinGW
platform were rebuilt and staged privately. Bounded 960x540 front/side captures
of the walk and joined transition pass and were visually inspected. Tide adds
a synthetic scene root, so preview edges bind authored bone names, not GLB node
indices. Preview follows the root's horizontal position; root travel and contact
quality still need inspection in the destination game. No engine file changed.

The runtime-only Windows archive is about 8 MB, including ONNX Runtime and
app-local Microsoft runtime DLLs, excluding models and development libraries.
tests/install.ps1 passes initial install, reinstall preserving model data,
spaces/apostrophes in the installation path, archive traversal/checksum refusals
without replacing the prior executable, and execution with only Windows system
directories on PATH. The test uses an isolated directory and changes no user
installation or user PATH. All selected upstream code/model/text licenses and
notices ship in licenses/; no sample motion data or research-only checkpoint is
part of the package. The GitHub repository is initially private.

Linux/macOS and optional Vulkan builds have not been run. General retargeting,
rigging, skinned character export, foot cleanup, collision constraints,
interactive editing and automatically chosen target placement remain absent.
The native dependency's constrained/multi-prompt APIs were not exposed then (see 0.3.0).

Packaging correction, 0.1.1: the installer now copies examples, notes and the
root license as well as the runtime and notices; the archive already held them.
The isolated installer test requires these files. Example output names have an
example- prefix so the supplied manifest does not collide with validation clips.
The tool is installed under this machine's LOCALAPPDATA/TideMotion, with its bin
added to the user PATH. Verified model files are hardlinked from the task's
download cache into its default model directory, retaining independent directory
entries without another 10 GB download/copy. No TIDE_HIDDEN variable was set in
the owner's environment.

## Public repository and automated releases (2026-10-27)

The owner authorized public distribution and platform release workflows. The
repository is now public at https://github.com/Tide-Engine/tide-motion. A scan of
every historical Git blob found no weights, evaluation motion data, oversized
binary payloads or common credential patterns. Apache-2.0 source and dependency
notices permit publication; selected NVIDIA motion-model terms permit commercial
use separately, and Meta Llama 3 retains its own terms and attribution. Models
remain external downloads, never Git or release artifacts.

Version 0.1.2 introduces .github/workflows/release.yml. Pushes to dev and pull
requests build Windows x64, Linux x64/ARM64 and Apple Silicon macOS natively.
Every job runs all three model-free CTests, packages only the runtime component,
writes a SHA-256 sidecar and exercises the installer against that archive. The
release job runs only for a v<project-version> tag after every build succeeds.
It verifies all four archives/checksums, creates or resumes a draft, uploads the
archives/sidecars/installers and publishes together. Published releases are not
overwritten on rerun. Actions and model/runtime dependencies remain pinned.
No manual workflow dispatch, publishing credential or model download is needed.

Archive names are tide-motion-<version>-<os>-<arch>.zip on Windows and .tar.gz on
Unix. Windows keeps app-local MS runtime DLLs. Unix packages carry ONNX libraries
with executable-relative RPATH; Linux still needs the distribution's compatible
glibc/libstdc++ (Ubuntu 24.04 is the release baseline). macOS 15 Apple Silicon is
the first Mac target. The Mac binary is unsigned/unnotarized. Intel Mac needs
an independently supplied compatible ONNX runtime; Windows ARM64 is not shipped.
Release builds use CPU inference, with AVX2 required on x64.

The new sh installer verifies the archive SHA-256, bounds archive paths and link
types, stages beneath the destination, tests the staged executable and replaces
bin with rollback on a failed move. It preserves models and prints a user PATH
entry rather than editing shell startup files. Its isolated test covers paths
with spaces/apostrophes, install/reinstall, model preservation, rejected checksum
and escaping symlink, and executing without loader-path environment variables.

Local validation: Windows x64 ClangCL Release and Ubuntu 24.04 x64 / GCC 13 under
WSL build successfully; all three CTests and both runtime archive/installer tests
pass. Two GCC misleading-indentation warnings were corrected without behavior
changes. actionlint 1.7.12 accepts the workflow. macOS and Linux ARM64 validation
is delegated to the workflow, not claimed as locally exercised. This packaging
change does not repeat the previously recorded 10 GB model/inference tests.
The Tide engine docs add a standalone tools/motion page and an animation-page
link with installation, batches, SOMA limits, asset import and license scope.

Initial CI correction: the windows-2025 label resolved to a VS 2026-only image,
which cannot configure the chosen VS 2022 generator. Use windows-2022 for that
toolchain. Homebrew LLVM 18 rejected fixed-enum declarations in Apple's SDK
headers reached by GGML/Accelerate; the Mac job uses Xcode's own clang/clang++.
Linux x64's complete build/tests/package/installer job passed on the first run.

Windows help text correction (0.1.3): the owner reported mojibake in the first
help line because its UTF-8 em dash was printed through a legacy console code
page. The banner now uses an ASCII hyphen; help text no longer depends on console
encoding for punctuation. This does not change console settings or argument/path
encoding. Windows Release rebuild, the existing cli-help check and an ASCII-only
output check pass. The rebuilt executable replaces the owner's installed copy;
model files and PATH are preserved.

## Mapped humanoid retargeting (0.2.0)

Owner decisions: retarget offline in Tide Motion; support humanoids through an
explicit reusable bone-mapping file, rather than choosing a single named rig.
No engine/runtime retargeting or added inference dependency is introduced.

`retarget SOURCE.glb --target CHARACTER.glb --map BONES.json -o CLIP.glb` converts
canonical SOMA motion to the character. `--to-soma` replaces `--target` for the
reverse conversion, enabling existing clips -> SOMA -> MotionBricks -> character.
Both directions also work as ordered batch actions. Mapping schemas and paths
are preflighted before inference; retargeting loads no model and standalone
retarget/inspect/preview no longer require a model-cache environment. Recipes
embed the actual mapping and record effective root scale/heading and discarded
non-root translations. Existing output/partial/recipe paths remain refused.

Mappings have SOMA names as keys and exact character names as values. Fifteen
anatomical slots are required (Hips, Chest, Head and three per arm/leg); other
SOMA slots are optional. Unique, nonempty rig/ancestor names and matching nearest
mapped ancestry are required, with extra unmapped helper/twist nodes supported.
examples/humanoid.json is an editable naming example, not automatic rig detection.

The reference rig is skin joints plus ancestors, or all nodes for an unskinned
skeleton file. Import orders parents first and checks cycles. Joint scale is a
static positive uniform scalar; world-pose evaluation composes ancestor scales.
Uniform affine matrices are decomposed; shear, reflection and nonuniform scale
are refused. Constant custom-rig scale tracks are accepted; changing scale,
cubic interpolation and morph tracks are not. Canonical model inputs retain
their stricter existing SOMA rules. Bounds remain 512 MiB, 1,024 nodes and 18,000
frames. inspect --skeleton lists names, parent indices, rest TRS and scales.

Hip separation defines world heading. Calibration aims primary mapped segments
along aligned SOMA reference directions, correcting A/T pose differences without
assuming identical local bone axes. Local XYZW referenceRotations overrides skip
automatic aiming for those character nodes. Per frame, the source world rotation
delta from its reference is conjugated by the source/target heading and applied
to the target reference, then converted through the current target parent. Local
target lengths stay fixed; unmapped nodes inherit parents with their calibrated
local reference transforms. This does not distribute roll among twist helpers.

Root travel scales by mean leg-length ratio, or explicit rootScale (the reverse
uses the reciprocal). Reference feet/toes establish ground, headingDegrees adds
yaw and groundOffset adjusts character-space height; all invert on roundtrip.
Root/ancestor source motion contributes through world-space evaluation. Non-root
translation animation is omitted and reported on stderr and in the recipe. This
is rotation/root-motion retargeting, not stretch transfer or contact IK.

Export retains the target hierarchy, rest transforms/scales and an animation;
it does not copy skin, mesh, materials or textures. asset.extras.tideMotion stores
the named motion root for following retargeted skeleton previews. Preview now
handles named custom skeletons, root-following and height-based framing. Tide
binds the animation to the original character through its existing APIs.

Validation: all five CTests pass on Windows x64 ClangCL Release and Ubuntu 24.04
x64/GCC 13 in WSL. New contracts use independently authored A-pose geometry,
nonuniform body proportions (not node scale), arbitrary bone axes, a rotated
centimetre armature and an unmapped twist helper. They check world limb
directions, standing ground/heading, root scaling, reverse position/rotation
roundtrip, manual reference overrides, optional bones, malformed/duplicate/crossed
mappings, matrix decomposition, constant/changing scale, preserved target lengths,
reported translation omission and GLB scale/root metadata roundtrip. CLI checks
exercise a two-direction batch with an empty model cache, absent cache environment,
manifest-relative paths with spaces/apostrophes, inspection and overwrite refusal.
Existing model-free motion and CLI checks pass; model algorithms are unchanged.
Windows and Linux runtime archives and isolated installers pass; actionlint and
the Tide docs production build pass. macOS/ARM64 remain native CI checks.

Real integration uses Khronos glTF Sample Assets' Cesium Man (Cesium, CC BY 4.0;
its logo also carries separate trademark terms), downloaded only to ignored
.research/retarget. No third-party model, texture or derived animation is committed
or packaged. A real 60-frame generated walk retargets to its different hierarchy,
and its original 61-frame animation converts to SOMA (with reported small non-root
translation omissions). An ordered batch makes an 8-token MotionBricks transition
to the generated walk and retargets the 145-frame joined sequence back, loading
MotionBricks once. Bounded native Tide captures of the custom skeleton and original
skinned character using the retargeted clip pass with no binding diagnostics and
were visually inspected. The skin and hierarchy are unchanged by this tool.

Remaining limits: manual bone assignment/calibration where needed, no foot locking,
collision constraints, automatic helper twist distribution or non-humanoid solver.
Retargeting does not guarantee production-quality contacts on arbitrary proportions.

## Windows GPU generation (0.2.0)

The owner asked for the GPU after a 2-second, 20-step CPU generation took about
13 minutes on their machine (RTX 4080 SUPER, AVX2 CPU). The Windows release is now
built with MOTION_VULKAN; the other platforms stay CPU-only until validated. CI
installs the Vulkan SDK 1.4.363.0 on the Windows runner from LunarG, pinned by
SHA-256, and the Windows job's timeout is 60 minutes for GGML's shader build.

Kimodo uses the first Vulkan device GGML selects. The CLI decides before loading:
it asks GGML for its device count (the same call Kimodo makes), and without a
loader, a driver or a supported device it says why and sets KIMODO_BACKEND=cpu.
A GPU that passes but fails to load Kimodo, or fails during generation (the text
encoder allocates then), is dropped: the model is reloaded on the CPU and the
same prompt and seed rerun; the rest of a batch stays on the CPU. On Windows
vulkan-1.dll is delay-loaded, so the CLI starts without it, and tide-motion is
compiled with /EHc- because GGML's C entry points throw, which /EHsc would let
escape the catch. KIMODO_BACKEND=cpu still forces the CPU. MotionBricks is
unchanged (ONNX Runtime CPU).

Validation on that machine: Vulkan and CPU-only Windows builds pass all five
CTests; dumpbin lists vulkan-1.dll as a delay-load import. A 2-second 20-step
clip took 35 s on the GPU against 778 s on the CPU, and a 4-second 50-step clip
27 s (whole command, including model hashing). Against the CPU clip of the same
prompt and seed, the GPU's largest differences were 1.6 mm and 0.004 in a
quaternion component; disabling Vulkan coopmat/F16 as Kimodo does on Unix was no
faster and no closer (19 mm), so Windows keeps GGML's defaults. Fallbacks were
exercised with GGML_VK_VISIBLE_DEVICES=9 (GGML throws), VK_DRIVER_FILES naming a
missing file (no driver), and GGML_VK_FORCE_MAX_BUFFER_SIZE=1024 (the GPU fails
during generation, then the CPU completes it). A machine without any Vulkan
loader was not available to test.

## Sequences, pose constraints and GIF previews (0.3.0)

Owner request: make the CLI a complete tool: multi-prompt sequences, pose
constraints (start/end poses, keyframes, effectors, paths), preview GIFs and
per-command help.

Kimodo: patches/kimodo-conditioning.patch is applied to the pinned kimodo.cpp
revision by FetchContent PATCH_COMMAND (cmake/apply-patch.cmake, idempotent via
git apply --reverse --check). It adds public model::generate_text_conditioned
(segments, transition, steps, seed, CFG, motion_condition{observed, mask,
first_heading}) and motion_dim(). observed/mask are raw [frames, 369] features
over the whole output. The pinned sequence sampler is generalised: segment 0 uses
the existing conditioned DDIM (or the unchanged unconditioned one when nothing is
masked); later segments add their cropped rows after the transition rows, moved
by the transition origin as upstream translate_2d. generate_text_sequence now
delegates with an empty condition (unchanged results). One unconditioned prompt
still uses generate_text.

Encoding is native C++ (src/constraints.cpp), adapted from NVIDIA Kimodo
(Apache-2.0, nv-tlabs/kimodo at e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954, only
read; licenses/NVIDIA-KIMODO-SOURCE.txt): kimodo_motionrep.py layout (smooth root
xyz, heading cos/sin from the hip line, root-relative joint positions, global
cont6d = first two matrix columns, velocities, foot contacts), constraints.py and
the _multiprompt transition sets. A full-body pose masks root xyz, heading, all
30 positions and the four end-effector base rotations (upstream's transition
adds that end-effector set to its full-body set); an effector masks its chain
positions (Hand+HandMiddleEnd, Foot+ToeBase) and base rotation with the root;
a path point masks root x/z and optionally heading. A keyframe's smooth root is
its hips (upstream substitutes the real root). A frame-0 constraint sets the
origin (subtracted before sampling, restored after) and first_heading; otherwise
heading 0 (+Z), as upstream. CFG stays 2/2 (upstream separated default).

Placement is a deliberate extension. Upstream refuses positions without the root
ground position. Kimodo starts at the origin, so an end pose from a clip that
also starts at the origin asked "walk forward" to return there: anchored, the
model walked 10 cm (pose error 3.3 cm mean). Leaving root x/z unmasked ("free")
kept the root-relative pose as accurate (1.6 cm / 7.0 deg vs 1.7 cm / 7.3 deg)
and the character travelled 84 cm. auto = world when frame 0 or a path fixes the
world, otherwise free and then the clip is translated so its last pose lands at
its file position. A pinned pose contradicting the prompt is missed more (10 cm
in the opt-in test before it was changed to free).

Correction (default, --no-correction / "correct": false): for each full-body
pose, per-joint local rotation deltas and the root delta (height only when free)
fade with a raised cosine over up to 10 frames either side, ending before any
neighbouring constraint frame. This replaces upstream's optional MotionCorrection
post-process (Eigen trajectory optimiser + IK + foot-skate cleanup), which was
not ported. Effector targets and path points are reported but not corrected (no
IK). The recipe stores modelError (before) and error (after) per constraint.

Validation (RTX 4080 SUPER, Vulkan build, 50 steps). An independent Node FK
checker reading the GLBs agreed with the printed numbers to 0.1 cm / 0.1 deg.
Model error, mean/max over joints, before correction:
- swing from guard.glb's last frame: root 0.8 cm, 1.3/3.9 cm, 3.5/8.6 deg; the
  same prompt unconstrained starts 22.7/59.1 cm, 46.2/97.5 deg away;
- swing and return (start + end, world): 1.4/3.9 cm, 5.3/19.2 deg and
  1.5/5.0 cm, 6.9/19.4 deg; two-prompt sequence from guard: 1.3/3.7 cm, 5.1/14.2;
- free end pose 1.6/4.2 cm, 7.0/18.7 deg; free keyframe at frame 60 1.3/3.0 cm,
  6.7/13.5 deg; right-hand effector 5.4 cm, 7.5 deg;
- path points: hips 5.9, 10.0 and 3.2 cm from the waypoints (the smoothed root,
  not the hips, is constrained); the final yaw -90 was met to 0.1 deg.
Rotation maxima are wrist/twist rotations positions do not determine. After
correction every full-body pose matched exactly (0.0 cm / 0.0 deg) and the
frame-to-frame motion near the seams changed by at most 0.2 cm. A nine-action
batch (sequences, all constraint kinds) took 67 s with one Kimodo load.
tests/inference.ps1 adds a two-prompt continuation from idle's last frame to
walk's first (free) and checks modelError < 6 cm mean and exact correction:
start 1.4/4.2 cm, end 1.4/5.0 cm; it passes with one load of each backend.

Model-free tests: encode a varied pose, decode with kimodo.cpp's own
decode_motion (identity statistics) and the native mirror, compare rotations and
root; encode(decode(x)) = x; heading convention; mask counts per kind; origin and
first heading; free/world placement; rejected out-of-range, duplicate-frame and
unknown-effector constraints; correction exactness and window. CLI contracts
cover per-command help, constraint/transition/placement refusals before any
model is checked, constrained batch manifests in a quoted path and the GIF
version probe. Vulkan and CPU Windows builds and Ubuntu GCC 13 (WSL, fresh
patched FetchContent) build without warnings and pass all five CTests.

Preview: the scene now samples before advancing the clock, so the first frame is
the clip's frame 0. --gif runs tide run --record with 2(F-1)+1 frames and
Playback.Clamp. Measured: GIF frame i equals the capture after 2i+1 frames
(PSNR 66.7 dB against 23 dB for neighbours), i.e. clip frame i; a 90-frame clip
gives a 90-frame GIF. The probe runs tide help run --record and asks for
0.5.0-nightly.45 or later when it fails. run_process gained optional output
redirection for it.

CLI: generate --then/--transition, --start-pose/--end-pose/--keyframe/--effector/
--path, --placement, --no-correction; batch prompts/transition/startPose/endPose/
keyframes/effectors/path/placement/correct; preview --gif; help COMMAND and
COMMAND --help. Version 0.3.0 (CMake project version, --version and recipes).

Left out: raw coordinate hand/foot targets (upstream effector sets need a whole
pose for rotation and root context, so targets come from SOMA clips); upstream
dense root paths with its smoothing (sparse waypoints only; many --path points
approximate one); IK for effectors and foot-skate cleanup; tunable CFG (added in 0.5.0); the
77-joint output; constraints on non-SOMA rigs without retarget --to-soma. Free
placement is validated empirically, not by upstream.

## Character previews and FBX input (0.4.0)

Owner request: preview on a character model rather than lines; a built-in
mannequin needing no third-party asset; FBX characters (bought packs are FBX).

FBX: ufbx v0.23.1 (MIT, licenses/UFBX.txt), pinned by FetchContent to
26a482ae66871d7de36eb722aa060bce95bce274, the revision Tide's compiler vendors,
and loaded with the options of Tide's importer (compiler/src/fbx_files.c):
right-handed Y-up, target_unit_meters 1, ADJUST_TRANSFORMS, geometry and inherit
helpers as nodes, clamped index errors, embedded content ignored. Node names,
parents and local transforms therefore equal what a Tide game imports, which is
what makes a retargeted clip bind to the game's FBX character. Alternatives
rejected: converting to GLB through another tool (an external dependency, and a
second interpretation of units/axes) and a hand-written parser. The rig is the
skin-cluster bones and their ancestors (every non-mesh node without skins); the
unnamed scene root is left out, as Tide ignores it when binding. Takes are
evaluated with ufbx_evaluate_transform at 30 Hz from time_begin; --clip selects
a take by name. Uniform static scale only, as for GLB; .fbx is detected by
extension in the GLB loaders, so inspect, retarget (--target and --to-soma
sources), batch and preview accept it. SOMA inputs to generate/connect stay GLB.
The Synty POLYGON character reads as 50 skin bones with Root scale 0.01.

Preview: --mannequin (default for SOMA clips), --character MODEL.glb|.fbx with
optional --map (retarget a SOMA clip first, reusing retarget()) and --texture,
--skeleton (the old lines; default for other rigs without --character). The
project declares `asset SkinnedMesh<Corner> body`, `asset Animation motion`
(the clip as GLB) and optionally `asset Texture skin`, samples into
body.defaultPose and draws with Draw.SkinnedMesh and a sun-and-sky frag, as
Parry draws its fighter (public documented APIs only). One camera (30 degrees)
sees two copies: facing it and in right profile, turned by the clip's
first-frame heading (SOMA hip line; 0 for other rigs). Each copy is centred on
the motion root above 5x5 floor tiles fixed in the world, so travel shows as the
floor sliding. Framing comes from the first frame's height; the floor is y=0 for
grounded clips (joints 2 cm above the soles otherwise sank the feet). Measured:
turn 180 shows the front (the character's right arm on screen left), -90 the
left side, so the camera's screen +x is world +x as documented.

Mannequin (src/mannequin.cpp): a GLB written here, SOMA-30 joints with their
canonical rest, inverse binds from the rest translations, one mesh of capsules
(14 sides, 5 cap rings) along each bone rigidly weighted to its parent bone, a
head and a nose; left limbs blue, right orange, centre clay; vertex colours.
Character renders ignore FBX vertex colours: Synty's are dark and blackened the
mesh when multiplied in.

Validation: Parry's PolygonSyntyCharacter.fbx (copied to scratch; not in Git)
with examples/synty-polygon.json (generic Synty bone names only) and its dummy
texture: guard.glb, parry-swing, parry-swing-return and sequence-from-guard
render on it and on the mannequin pose for pose. Frames were inspected: no
inverted limbs or twisted wrists, feet on the floor, the mesh follows the bones,
hands and fingers intact. The guard's apparently sideways mannequin head was
checked numerically: hips and chest face +Z, the head is turned 55 degrees in
the clip itself. The Synty sword idle FBX plays directly on the character and
converts with --to-soma (non-root translations are reported and dropped); its
SOMA version leans further forward than the FBX, a calibration gap of
--to-soma on that rig, not of preview.

Tests: motion-tests checks the mannequin GLB (rig reads back as canonical SOMA,
rigid weights to valid joints, unit normals, indices in range, positive
enclosed volume, i.e. outward glTF winding). retarget-tests writes its
independent centimetre test rig as an ASCII FBX turned to Z-up and unscaled
(no third-party content), reads it back in Y-up metres with identical world
positions (1e-4) and identical retarget results, and checks the no-take and
SOMA-must-be-GLB refusals. The CLI tests inspect and retarget onto that FBX and
write preview projects for all three bodies (with a stand-in Tide), plus the
option refusals. Both Windows builds and Ubuntu GCC 13 pass.

Gaps: one take per preview; no materials beyond one optional texture (no
normal maps, multiple materials or embedded textures); no automatic mapping;
FBX scale animation and non-uniform scale are refused; preview needs the clip
and character to share names when --map is not used; --to-soma from Synty
leans the spine (see above).

## Generation controls (0.5.0)

Owner request (2026-10-28): pick quickly from several variants and expose every
useful control. About 100 kick clips generated that day varied a lot by seed,
prompt wording and duration ("A person kicks forward with the right leg" kicked
29/35 across 1.5–5 s, a long sword-stance prompt 15/35, kicks mostly vanished
beyond 3 s).

CLI (generate): `--variants N` (1..64, seeds S..S+N-1 with one model load;
outputs CLIP.seedS.glb plus recipes, the plain name only when N is 1),
`--variants-gif`, `--tide PATH`, `--guidance W` and `--constraint-guidance W`
(Kimodo's separated text/constraint CFG weights, 0..10, finite, default 2/2 as
before), `--device auto|gpu|cpu`, `--threads N` (1..1024), `--correction-frames N`
(0..60, default 10), `--in-place`. connect: `--threads N` (ONNX Runtime
intra-op threads; default stays half the hardware threads). New command
`mirror CLIP.glb -o OUT.glb [--clip NAME]`. Batch fields: variants, variantsGif,
tide, guidance, constraintGuidance, device, threads, correctionFrames, inPlace;
connect threads; a mirror action (source, clip, output). Batch preflight expands
variant names and sheet paths and refuses any collision before anything runs.
Recipes add guidance, constraintGuidance, correctionFrames, inPlace, device (the
one actually used), threads and variants {count, firstSeed, sheet}; mirror
writes a recipe too. An unknown command (also `help NAME`) now says so instead of
printing the whole help. Every new value is validated before models are checked.

Device: auto is the 0.2.0 behaviour (probe, CPU fallback on probe/load/generate
failure; later auto actions of a batch stay on the CPU after a failure). gpu
probes Vulkan and fails instead of falling back (a CPU-only build refuses before
hashing the models); cpu sets KIMODO_BACKEND=cpu without probing. Threads set
KIMODO_THREADS, which kimodo.cpp reads when its CPU backends initialise
(motion weights and text encoder); 0 restores the operator's value. Both apply
at load, so a batch action asking for a different device/threads reloads Kimodo
(the operator's KIMODO_BACKEND/KIMODO_THREADS are remembered and restored for
auto). kimodo_runtime_options exists in the C API but is not wired to the C++
model::load in this revision, so the environment is the interface.

Variant contact sheet: platform.cpp writes one Tide project with the mannequin
and one Animation asset per clip, drawn through an orthographic camera
(float4x4.Ortho) so cells have no perspective; Draw.SetScreen text labels "seed
S" and each pose's time. A cell shows six key frames (edit.cpp key_frames: the
first and last frames, then local peaks of the largest root-relative joint
movement from frame 0, kept at least frames/12 apart, then midpoints of the
widest gaps), right profile above, facing below, a floor line under each row.
About 960 pixels a column, at most 4096 either side; 8 variants are 1920x2264
and render in about 3 s with `tide run --capture --frames 3`. The GIF lays each
variant out facing and in profile and records the longest clip once. Any sheet
failure (Tide absent, old Tide) is reported as skipped after the clips are
saved. Checked on a real 8-variant kick run: the one clip kickcheck scored as no
kick (seed 5, foot 0.03 m) is plainly standing in every cell, the seven kicks
(0.58–1.25 m) show the leg up in profile.

0.5.1 (2026-10-28): cells were sized from the first clip's first frame, so a clip
that starts crouched (a state prompt holds its state from frame 0) or jumps or
reaches out spilled into its neighbours. Cells now fit the tallest pose of any
clip (every second frame, plus 0.15 m for the mannequin's flesh beyond its
joints) and space poses at least 1.5 times the widest horizontal reach from the
root plus that margin. Checked on 8 crouch, 4 kick and 4 jump variants: no
overlap; standing kicks look as before.

## Prompting guide (0.5.1)

PROMPTING.md (packaged beside README.md) records what to write and how to nudge
a result, from measured runs on 2026-10-28: 17 prompt patterns and 4 crouch
transitions at 8 seeds each, a duration sweep (1.5–5 s, 5 seeds), and steps
10/25/50/100 at 8 seeds on two prompts, scored by forward kinematics (foot peak,
both feet airborne, head drop, root travel). Findings worth keeping in mind when
changing generation: wording style is irrelevant; side and direction words are
obeyed ("forward" travels about 1.6 m); props weaken the action; a two-action
prompt fails where --then succeeds; a state prompt ("crouches down") holds the
state from frame 0, a transition needs "a standing person …" or a start pose;
steps above 50 do not help and a seed's result depends on the step count. The
README's measurement tables moved there.

Mirror (edit.cpp mirror_soma): SOMA-30 rest offsets are symmetric about x
(left is +x facing +z; largest asymmetry 0.24 mm at the hips) with identity rest
rotations, so a local rotation L becomes M L M (M the x reflection, quaternion
(x,-y,-z,w)) on its Left/Right counterpart; the root takes A^-1 M A W M with A the
first frame's heading, and its position is reflected across the plane through
the first frame's root, up and its facing. The clip starts in the same place and
facing (within 0.3 degrees, from that asymmetry); mirroring twice returns it to
within a few millimetres. A Kimodo right kick (right foot 1.25 m, left 0.01 m)
mirrors to left 1.24 m, right 0.01 m.

In place (remove_root_travel): the hips' ground path averaged over 31 frames
(ends padded by point reflection so steady travel stays linear) is subtracted
from every frame relative to frame 0; height, facing and sway stay. A 3-second
walk that travelled 4.85 m stays within 0.11 m. Applied after correction;
recorded constraint errors precede it. Refused with --path (the path only
specifies travel). Not foot planting: feet may slide where the averaged travel
differs from the true contact motion.

Correction window: correct_poses takes the frame count (0 = only the frame).
The default output is byte-identical to 0.4.0 (same seed, prompt and poses).
posecheck on a 2.5 s kick between idle poses: model error 60 degrees at the
hands; window 0 pops 60.4/60.2 degrees in one frame at the ends, 2 frames 30,
4 frames 21.6; 10 (default) and 20 leave the kick's own 30-degree steps as the
largest.

### Measurements (RTX 4080 SUPER, Vulkan, 50 steps)

Kicks start and end in Parry's idle (`--start-pose idle-soma.glb --end-pose
idle-soma.glb@0`); scored by forward kinematics (scratch tools score.py,
kickcheck.py): kick = a foot 0.45 m above the floor; slide = mean horizontal
movement of a foot within 3 cm of the floor (cm/frame); pose error = recipe
modelError positionMean (max) at the two poses before correction. Batches with
variants ran at about 1.5 s a clip after loading (160 clips plus 20 sheets in
4 min 12 s).

Text guidance, short prompt (24 seeds at 2/3/4, 8 at 1/6), kicks 2.5 s / 4 s:
1: 0/8, 0/8; 2: 18/24, 12/24; 3: 15/24, 21/24; 4: 23/24, 23/24; 6: 8/8, 8/8.
Slide 0.15/0.06, 0.22/0.15, 0.24/0.21, 0.29/0.29, 0.39/0.45. Pose error
2.1/2.0, 2.1/2.1, 2.1/2.4, 2.3/2.8, 2.4/4.1 cm (max 4.9 to 10.0). Sword prompt
(8 seeds): 2.5 s 1, 6, 7, 8, 8 kicks of 8; 4 s none at any guidance. Sheets:
guidance 4 kicks reliably but adds a boxer's guard to most variants; 6
caricatures (hops, arms flung up). Default kept at 2; documented 3–4 when the
move is missing. The worst single per-frame joint step (87 degrees) was one
sword clip at guidance 3; otherwise worst steps rose with guidance from about 13
to 35 degrees, mostly the kick itself.

Constraint guidance, short prompt, 8 seeds, kicks 2.5 s / 4 s and pose error
(cm mean/max): 1: 8/8, 5/8, 5.5/12.9 and 3.7/9.2; 2: 7/8, 2/8, 2.1/5.1 and
2.1/5.0; 3: 7/8, 1/8, 2.2/5.2 and 2.1/4.7; 4: 7/8, 1/8, 2.3/5.4 and 2.1/4.8;
6: 6/8, 0/8, 2.9/6.0 and 2.4/5.6. 2 is the closest overall; higher also pulls
the motion toward the idle constraints and suppresses the kick. Default kept.

Negative prompt (tried, not shipped): a patch let a negative-prompt embedding
replace the zero text of both non-text separated-CFG branches (constraint and
unconditional; with no constraints that is standard negative CFG,
n + w(t - n)). Same 8 seeds, guidance 2, kicks (baseline / "standing still" /
"walking" / "A person stands still doing nothing"): short 2.5 s 7 / 7 / 8 / 8;
short 4 s 2 / 3 / 0 / 7; sword 2.5 s 6 / 5 / 1 / 7; sword 4 s 0 / 0 / 0 / 0. Only
the full sentence helped, about as much as guidance 3 (8/8 on the same seeds),
whose sheet looked more natural (clean front kicks rather than knee chambers
with a guard); "walking" removed kicks; one clip popped 96 degrees. Not clearly
better than guidance, so the patch and --avoid were removed.

Validation: model-free contracts for variant/sheet names, mirror (twice, start
place and facing, left/right swap against the reflected FK), in place (travel
removed, frame 0/height/facing/sway kept), key frames (span, peak included) and
the correction window (0 and 3); CLI contracts for every new range/refusal, the
unknown command, variant collisions in CLI and batch, mirror CLI/batch and per-
command help. Both Windows builds (Vulkan and CPU, ClangCL, warnings as errors)
pass all five CTests. Not measured: --threads timing on the CPU path (only
validated and plumbed), CPU fallback of --device gpu on a broken driver (the
refusal path is the CPU-only build's), Linux/macOS builds (left to CI).

## Offline MotionBricks pose authoring (2026-10-28)

`pose-sequence POSES.json -o OUT.glb` and the matching batch action add bounded
explicit pose-target authoring. Design references read at the unchanged pinned
MotionBricks revision 54fc563aeaaf819a753d72cc9fdc6713989cdd43: docs/inbetweener.md
(ordered targets, sampled windows, chaining recent context) and
 docs/smart_primitive_editor.md (authored poses placed around interaction objects).
Only source/documentation was inspected; no upstream sample/evaluation motions
were copied. No model, dependency or license revision changed.

The versioned JSON schema, example and full field semantics are in README.md and
examples/pose-sequence.json. Absolute Y-up hips position and absolute hip-derived
facing place a sampled SOMA pose without changing local body rotations. Start
uses the preceding four 30 Hz samples ending at the chosen frame; targets use
the following four beginning there. Sampling clamps at source boundaries.
Optional hold repeats the pose four times, giving zero guide derivatives under
the existing four-pose contract (not upstream's one-pose mask mode). A later
segment uses the last four generated samples as its context. Alignment, masks,
trim and native endpoint correction remain unchanged in bricks.cpp; no alternative
inference, blending generator or scene simulation was introduced.

Each target requires 6..16 tokens. A trimmed bridge contains 4*tokens-6 samples;
exporting only one seam copy adds 4*tokens-7 intervals. The output starts at the
selected start pose, without the three context lead-in samples or clip tails.
Arrival frame/time is exact; arbitrary second durations and early transitions
are intentionally absent. Sample normalization can change a seam quaternion by
roundoff: the already exported endpoint is copied back bit for bit before append.
Append refuses a nonexact seam. Effective recipes record absolute input paths,
resolved source frames, placements, hold and token settings, computed arrivals,
threads, version, model source revision and output count. Their authoring object
can be extracted and rerun; derived timing fields are recomputed.

All targets/GLBs are resolved and validated before any session is loaded. Bounds:
1 MiB recipe, 128 targets, existing GLB/frame limits. Batch preflight loads every
pose-sequence recipe before any action executes, and the existing global output/
partial/recipe collision checks cover its outputs. Sources must already exist
at preflight, so pose-sequence cannot consume a clip created by an earlier action
in that same batch. Windows are retained, not full source clips. One RunContext
owns/reuses the ONNX session across sequence segments and interleaved connect
actions with the same threads setting; connect behavior is unchanged.

Validation on Windows x64 ClangCL 18 Release: all five CTests pass. New independent
model-free checks use a translating authored rig to verify absolute placement,
facing, rotated travel, last/first four sampling, held guides, negative source
indices, unchanged local body rotations, accumulated arrival frames, reusable
effective recipes, exact seam omission/refusal, fractional timing and late bad
sources. CLI tests reject a late missing source in batch before the first generate
loads models, plus reserved output/partial/recipe paths and per-command help.

Real native ONNX CPU inference (8 threads), coordinated with Windows named mutex
TideMotionGpuValidation, used existing locally Kimodo-generated idle/walk clips.
Two targets (walk frame 15 at [0,0.95,0.8], yaw 0, 8 tokens; idle frame 0 at
[0.5,0.95,1.4], yaw 90, 10 tokens) produced 59 frames, arrivals 25 and 58.
A following ordinary 8-token connect produced 26 frames; the two-action log has
one model verification and one ONNX session load. Native exported-GLB verifier
checks every local endpoint translation within 1e-5 m and rotation within 0.002
rad, plus all arrival/frame counts. All pass. Reproducible opt-in runner:
tests/pose-inference.ps1, taking models and user/self-generated idle/walk inputs;
it acquires the shared mutex, checks session reuse/endpoints and optionally previews.
No weights or generated/third-party motion assets enter Git.

Bounded TIDE_HIDDEN=1 mannequin capture at frame budget 51 succeeds and was
visually inspected (walk pose, facing/profile, no binding diagnostics). Tide CLI
and MinGW platform were built from D:/Projects/PurrEngine at
28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c and staged privately under this task's
.research. Initial staging lacked native resources; installing the matching
platform/CLI components resolved it. No engine changes were made. The capture
is functional asset evidence, not a continuous contact-quality assessment.

Limits: exact poses do not guarantee velocity/acceleration continuity, plausible
travel between contradictory/distant targets, contact preservation, collision
avoidance or artist-ready motion. No scene/object understanding, pivots, automatic
placement, IK/contact cleanup or Kimodo v1.2 work is included. Only explicit SOMA
poses; mapped humanoids first use retarget --to-soma. Linux/macOS and GPU timings
were not tested for this feature; MotionBricks remains native CPU ONNX.

Integration: rebased onto current origin/dev, preserving the concurrent 0.5.1
prompting/preview/install updates and ARDY research. Rebuilt and all five CTests
pass again; exported endpoint/timing verification and the bounded Tide capture
also pass with the integrated executable. A repeat via the opt-in inference
runner correctly refused while another thread held TideMotionGpuValidation;
no conflicting inference was started. Real inference evidence above is from the
successful earlier mutex-protected run, with author.cpp/bricks.cpp unchanged.

## Explicit native contact IK (2026-10-28)

Investigated nv-tlabs/kimodo MotionCorrection at the pinned upstream revision
`e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954` (read-only checkout in ignored
.research/kimodo). Its CMakeLists defines motion_correction_cpp_base but requires
Python3 Interpreter/Development and pybind11 unconditionally, fetches Eigen 3.4.0,
and adds /arch:AVX or -msse4.1/-mavx. More than build flags need adapting: Math/SIMD.h
includes immintrin.h and uses __m128 throughout. TrajectoryCorrector.cpp solves
Eigen sparse trajectory systems; InverseKinematics.cpp implements geometric
TwoBoneIk. The source is Apache-2.0 with NVIDIA notices. Existing source/model
licenses remain in licenses; no upstream files, Eigen, bindings or weights were
added. This is not a port of its trajectory solver or evidence of numerical parity.

Chosen equivalent for a deliberately smaller supported scope: independently written
scalar C++ geometric two-bone IK in src/cleanup.cpp (tide-two-bone-v1). Uses the
cosine rule and current knee plane, with a deterministic fallback plane at straight
knees, world-to-local rotation correction and preserved world ankle orientation.
It adds no runtime/library dependency, architecture flags, Python or service.
Existing dependency pins are unchanged. The scalar implementation has no x86
intrinsics; ARM64/Linux/macOS have not been built locally in this change.

`cleanup SOURCE.glb|.fbx --contacts PLAN.json -o OUT.glb [--map BONES.json]
[--clip NAME]` is a model-free, optional separate postprocess. Batch `cleanup`
fields are source, contacts, output, optional map/clip. Put it after generate,
connect or retarget. Their existing defaults and inference remain unchanged.
A strict <=64 KiB plan has 1..1000 explicit inclusive contacts
{foot: left-foot|right-foot, start, end}, each at least three frames, zero-based,
0..17999; same-foot overlaps refused. Each anchor is its first ankle position,
with no ground estimation/projection or automatic contact classification.
fadeFrames 0..60 defaults to 3, maxCorrection metres (0,0.5] defaults to 0.1.
The latter rejects a whole interval whose ankle travels farther than that from
its anchor. Raised cosine target fades at interval edges/protected poses leave
end frames unchanged with nonzero fade. If solving increases interval mean
horizontal ankle movement by >1e-7 m/frame, restore its entire original chain.
Skipped intervals are saved and reported (correction-bound/drift-increased).
This bounds displacement, not joint angles; max local angular change is reported.

SOMA-30 or explicitly mapped humanoids only. Requires two distinct direct
thigh/calf/ankle chains beneath the motion root; intermediate helpers there are
refused (helpers elsewhere can remain). FK respects ancestor uniform scales.
Every local translation, root rotation/path and world ankle orientation is kept.
Unreachable targets leave reported error rather than stretching/moving the root.
protectedFrames holds exact whole poses. Constraint frames from the adjacent
source recipe are protected automatically and retained for repeated cleanup;
external clips whose constraint metadata was not transferred need explicit
protectedFrames. Retarget carries known constraint/protected frame indices into
its recipe; pose-sequence start/arrival frames and previous protectedFrames are
also inherited by cleanup. No assumption that an ankle is a sole or toe.
Recipes include effective intervals/settings/mapping, solver version, source
recipe, protected frames, mean horizontal step before/after (m/frame), max anchor
residual (m), max local joint correction (degrees), samples and skip reason.
Metrics include fades/protected samples; residual is distance to the anchor,
not ground clearance. Bone lengths are preserved; this does not enforce anatomical
joint limits, collisions, pelvis balance or distribute twist. Toe articulation
remains as authored. Explicit contacts must exclude kicks, flight and pivots.
Generation's --effector and --path targets remain measured, not solved by this
postprocess; no general full-body trajectory optimiser or hand IK is claimed.

Validation: Windows x64 ClangCL 18.1.8 Release, warnings as errors for project
code, all six CTests pass. New native independent planar geometry contracts check
analytically known ankle targets, double support, length/root/translation/world
ankle rotation invariants, protected poses, interval boundaries, correction-bound
skip, increased-drift rollback, centimetre ancestor scale, unreachable targets,
bad intervals and GLB roundtrip. CLI tests exercise cleanup/batch with no models,
quoted paths, source recipe protection/reporting, overwrite refusal, numeric
refusal and full plan preflight before an earlier inference action can run.

Real asset validation reused existing native generated assets, with no new
inference, GPU timings or weights. Optional developer-only
`python tests/verify_cleanup.py BEFORE.glb AFTER.glb LEFT_ANKLE RIGHT_ANKLE`
independently decodes dense Tide GLB tracks and implements FK (including scales),
asserts reported drift, unchanged translations/Hips, preserved world ankle
orientation, protected pose equality and bone lengths. Python is not used by
cleanup, inference, the runtime, build or the six CTests. Native
`cleanup-tests BEFORE.glb AFTER.glb` additionally checks root, all local
translations and bone lengths after export.

Real idle: D:/Projects/tide-motion/.research/inference-dae12e915f8d42b6919e3777c94a5828/idle.glb
(60 frames, seed 1, 50 steps), both feet explicitly planted 0..59, default fades
and correction bound. Mean ankle horizontal movement left/right, mm/frame:
0.273/0.261 -> 0.062/0.153. Max anchor residual including fades 4.46/8.27 mm;
max local correction 2.10/1.85 degrees. The same clip retargeted with
examples/synty-polygon.json onto scratch PolygonSyntyCharacter.fbx:
0.301/0.332 -> 0.078/0.231 mm/frame; residual 6.67/13.14 mm, correction
4.06/2.50 degrees. Native checks: world root/local translation error zero;
max world bone-length change <2.4e-7 m. Independent FK checks <1.4e-7 m.
These are modest low-drift idle examples, not broad locomotion-quality evidence.

Real parry-swing-return (75 frames, contacts both feet 0..15): original exact
constraints at 0 and 74 stay within 2e-6 per exported component, zero recorded
constraint errors preserved. Left ankle mean 0.787 -> 0.672 mm/frame. Right would
increase from 1.326 to 1.397 due to the fade/reach limits: now restored to 1.326
and reported drift-increased. Real fix-kick.seed3 (75 frames), deliberately
mislabelled left-foot 5..60: exceeds the 10 cm bound (1.681 m anchor distance),
skipped with zero joint correction; mean 33.154 mm/frame unchanged. Protected
frame 15 unchanged. This demonstrates refusal of that mislabelled kick, not an
automatic airborne detector. Validation plans/assets/captures remain ignored in
.research/validation; no third-party character or derived motion is distributed.

Visual validation: rebuilt CLI/engine and MinGW engine/platform from
D:/Projects/PurrEngine at 28b61e4db, staged privately under .research/engine (no
engine source edits). Packaging install stopped on the unrelated unbuilt editor,
after staging required CLI/engine/platform files; the native CLI was restored
after the MinGW staging pass. TIDE_HIDDEN=1, 30-frame mannequin and 60-frame Synty
captures at 960x540 succeeded and were visually inspected: feet at the floor,
no inverted knees/limbs or broken skin binding in those sampled poses. The idle's
forward lean remains. This is limited sampled visual inspection, not a guarantee
of temporal quality or full foot locking. No GPU inference ran, so no GPU mutex
was needed; future inference validation must hold TideMotionGpuValidation.


Integration completion: rebased onto origin/dev c714588, retaining the concurrent
pose-sequence implementation, 0.5.1 prompting/preview/installer changes and ARDY
research. Resolved additive shared-file conflicts; rebuilt the integrated CPU
executable and all six CTests pass. Added coverage for inherited protectedFrames,
pose-sequence arrivals and protection through two retarget directions; both
affected CLI suites pass. The new native test uses a unique temporary filename,
and integration CTests used .research/test-temp as TEMP/TMP to isolate this
worktree from concurrent tests.

A final real constrained generate -> Synty retarget -> cleanup run preserves
frames 0/74 through metadata propagation. Independent FK: left/right ankle mean
1.644/0.936 -> 1.359/0.768 mm/frame on intervals 0..15, world bone-length error
<1.3e-7 m. Native checker: root/local translation error zero, world bone-length
error <1.8e-7 m. The independent checker verifies every protected frame component
within 2e-6 and ankle orientation. No model load/inference was repeated for this
integration; existing generated assets were consumed. No PR, model publication,
force push or additional workflow dispatch is part of this work.
## DART / DartControl source-only investigation (2026-10-28)

Detailed evidence, exact release artifact IDs/configuration hashes, source pins,
native contracts and follow-up gates are in [dart-feasibility.md](dart-feasibility.md).
Source revision: zkf1997/DART bb67ae1ed6ce051080468bf15bc6e54a6c3f8417;
Apache source notice retained in licenses/DART-SOURCE.txt. The author explicitly
confirms Apache-2.0 for checkpoints in public issue 25, independently of the
repository badge. This is positive checkpoint distribution/inference evidence.
AMASS dataset access/training restrictions are a separate question and do not
by themselves prove that downstream checkpoint inference is prohibited.

Unresolved exact runtime assets: the release's external
data/seq_data_zero_male/mean_std_h2_f8.pkl needs a commercial inference/conversion
grant; a compatible zero-beta male SMPL-X skeleton/calibration artifact with a
permissive grant has not been identified. Stock headless inference constructs
restricted full SMPL-X body models. Predicted-joint history can avoid per-chunk
body inference, but still needs the exact pelvis calibration offset and initial
history encoding. The CC BY SMPL-X Body subset is a possible route, not evidence
that the full parametric NPZ or an arbitrary extracted offset is cleared. No
body model, weights or data bundle was fetched for inference, no one contacted,
and no placeholder backend, install entry or CLI promise was introduced.

Forward continuation and the separately trained goal policy require no
test-time gradients. The policy drives initial diffusion noise and has specific
walk/run/left-hop coverage, a clipped 1069-value observation and a 0.3 m success
threshold. Its checkpoint selects a different denoiser from the continuation
demo. Inbetweening instead optimizes initial noise with Adam through every DDIM
step, decoder and autoregressive coordinate/history transform. Native gradients
remain separate work and do not preclude forward-only implementation once the
asset gate clears. DART's 276 features use row-6D rotations, Z-up coordinates,
two history/eight future frames; canonical SOMA conversion/mapping, parity and
real goal/GLB/preview/packaging tests are not yet implemented or measured.

Lawful native proof work: tests/dart-feasibility is an isolated C++23 project,
with no production dependency/build changes or installed artifacts. Windows x64
ClangCL 18.1.8 Release passes both CTests: independently specified row-6D,
degenerate rotations, pelvis/world inverse and handedness fixtures; six local
x0-DDIM adjoints match finite differences to 1.67395e-11 maximum error. An
in-memory synthetic ONNX opset-20 MatMul/LayerNormalization/Gelu graph on pinned
ORT 1.31.1 CPU matches independent scalar results to 2.37287e-08. The inference
DLL returns no training API. This is operator/math evidence only, not DART
export, network gradients, SOMA mapping or inference parity. No GPU used; no
inference time/memory, goal/seam/action/foot quality numbers are fabricated.
TIDE_HIDDEN=1 was set for validation. Existing backends/defaults and concurrent
ARDY integration files are preserved. Reproduce with the isolated commands in
the detailed note; use a thread-specific build/temp directory.


## Opt-in smooth-root SOMA-RP v1.2 (2026-10-28)

Baseline: v0.5.0 / dd4dc0b. Upstream source inspected at
nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 (CHANGELOG.md,
kimodo/motion_rep/reps/kimodo_motionrep.py). The October 15 release corrects
training/test overlap; that correction alone is not evidence of quality gain.
This work uses only nvidia/Kimodo-SOMA-RP-v1.2, never the distinct ArdyRep variant.
No evaluation dataset or research-only checkpoint was downloaded.

Pre-download inspection: official HF revision
0e3c89024d896064b6851201373f7834784a386b, config.yaml and LICENSE, six small
statistics arrays and the first 65,536 bytes of the safetensors file. LICENSE
SHA-256 11ccdcb47db87ac30b63765b5e5154403026c4fbb52c19919d99d912ec454e19
matches the existing NVIDIA-KIMODO-MODEL.txt terms after
line-ending normalization. Its license permits commercial use subject to its
terms; the existing NVIDIA notice and separate Meta text encoder terms remain.
Config SHA-256 82f1e9f44d0df517d8ebde03d2e9396fd0bf372360714757901c1d97a2626599.
Architecture: two-stage transformer, 1024 hidden, 2048 FF, 16 layers, 8 heads,
GELU, 50 text tokens, 4096 text width, separated CFG, 1000 base steps, 30 Hz.
Header: 408 F32 tensors, 283,281,777 parameters; root input/output widths 738/5,
body 737/364. Feature order: smooth-root xyz (3), heading cos/sin (2), 30 joint
positions (90), global first-two-column rotations (180), velocities (90), four
contacts: 369 total. Local root has angular velocity, horizontal velocity and
height (4). Statistics are C-order little-endian F64 vectors of 5/4/364 values,
all finite, all standard deviations positive. They are converted to F32 and use
the existing sqrt(std^2 + 1e-5) normalization. No v1.1 statistics are substituted.

The complete checkpoint then downloaded privately and matched its upstream LFS
SHA-256 fdcef624140a12c0a4840192ce0b75cafb4e73228a893da3864b4a990d3e919e
(1,133,177,724 bytes). Conversion changes only the pinned converter's model
identity allowlist; no tensor transform or inference patch is needed. The
414-tensor F32 GGUF is 1,133,166,784 bytes, SHA-256
9c24c5ee9b6b3683947ab2796727ffb2fbd69b2ebc7683214809013687d87a19.
All source/statistics hashes are pinned in tools/soma-rp-v1.2.json. The native
kimodo.cpp/GGML revisions and conditioning patch stay unchanged.

Contract: generate --model v1.1|v1.2; default v1.1. Batch field model accepts the
same values, fully preflighted. Recipes add model and modelGgufSha256, plus
checkpointRevision for v1.2. A model switch reloads Kimodo; checksums are checked
once per selected version per process. v1.1 retains kimodo.gguf; v1.2 uses
kimodo-v1.2.gguf; both share the existing pinned text bundle. models install/check
kimodo-v1.2 are explicit opt-ins; all retains its prior selection. There is no published pinned v1.2 GGUF found in the HF model search at inspection
time. Installation therefore downloads the official safetensors, config, license
and statistics to .soma-v1.2, verifies all pins before parsing, writes the same
GGUF natively in C++ and verifies its full pinned digest before rename. Verified
sources remain cached (about 1.13 GB extra disk space). No user Python is needed.
The Python stdlib/curl preparation tool is an independent development reference;
it is not packaged or invoked by shipped inference/installation. Native input
parsing is intentionally restricted to the exact pinned files, never arbitrary
user safetensors or checkpoints. Dependencies and model licenses stay unchanged.

Validation (Windows x64 ClangCL 18.1.8 Release, Vulkan, RTX 4080 SUPER): all five
native CTests pass with TEMP/TMP isolated beneath this worktree. The first run
hit the known shared-temp collision; its isolated rerun passed. Three development
conversion tests pass (license/input refusal, F64-to-F32 statistics and truncated
input rejection). Native conversion of the verified official files produces a
byte-identical pinned GGUF to the independent Python reference. Native install
also downloaded missing LICENSE/config via HTTPS, retained/verified cached
weights/statistics, converted, verified and installed shared text successfully.

The reproducible tests/soma-v12.ps1 batch makes 31 GLBs: per version, idle (2 s),
walk (3 s), right kick (2.5 and 4 s), each with seeds 1/2/3; a 2.5 s kick between
the same generated v1.1 idle poses; a 3 s walk with three path points; a 1.5 s
walk -> 1.5 s right-hand wave starting from that idle. One final v1.1 repeat
checks switching back. All use 50 steps and default CFG/correction. Exactly three
Kimodo loads occur; every GLB imports, every recipe identifies the selected
checkpoint, and the repeat's binary animation payload equals the first v1.1
idle after switching versions (GLB names differ). A sampler Int32 overflow in the
first harness was fixed to Int64; resource/timing evidence comes from its fully
completed rerun holding TideMotionGpuValidation throughout. The installed owner's
exp2.json batch was observed concurrently outside this mutex and left alone;
these timings are contended, not isolated benchmarks.

Independent GLB FK (tests/soma_metrics.py): each version passes all 3 idle travel
proxies (<30 cm), all 3 walk travel proxies (>50 cm), and all 6 kick proxies (foot
height >45 cm). These coarse proxies do not establish semantic precision or
artist readiness. Foot-slide proxy = mean horizontal cm/frame of feet/toes when
both adjacent positions are below 3 cm, averaged per clip and then across seeds:

| Case | v1.1 | v1.2 |
| --- | ---: | ---: |
| idle, 2 s | 0.026 | 0.025 |
| walk, 3 s | 0.117 | 0.135 |
| kick, 2.5 s | 0.181 | 0.161 |
| kick, 4 s | 0.072 | 0.087 |

This proxy differs from upstream benchmark metrics and can omit airborne feet.
Before correction, start/end pose joint errors (mean/max cm) were v1.1
1.30/3.47 and 1.56/4.49; v1.2 1.05/3.29 and 2.09/5.39. Mean rotation errors
were 5.76/5.59 degrees vs 5.15/4.81. Correction makes both poses exact. Path
hips-ground errors at frames 0/45/89: 5.04/5.38/7.01 cm vs 5.68/4.21/4.54 cm;
the constrained feature is smooth root, so hips errors are reported explicitly.
The multi-prompt start errors were 1.32/4.09 vs 1.24/3.44 cm; both sequences
raise the right hand above the head in the second half (24.1 vs 17.9 cm peak).

Contended DDIM sampling medians over the 12 unconditioned clips: v1.1 1.499 s
(range 1.112..3.049), v1.2 1.162 s (0.939..1.351), excluding text/load/decode.
Sequential order and the external workload prevent attributing that difference
to the checkpoint. Total 31-action process: 102.76 s; sampled peak host working
set 4,212,203,520 bytes. Both checkpoints have equal F32 weight size and exactly
the same observed graph scratch allocation sizes (maximum 14.80 MiB per graph).
Per-checkpoint peak host/VRAM usage was not isolated; WDDM and another inference
process preclude interpreting whole-device VRAM as this process's allocation.
No speed or memory improvement is claimed.

Bounded hidden Tide previews use D:/Projects/PurrEngine at
28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c: rebuilt CLI target and MinGW platform,
staged privately with its engine files (the broad install's unrelated missing
editor target was bypassed by staging the platform component). Six mannequin
captures at 90/179 frames were inspected: right kicks, walking first segment,
and right hand raised in the second segment, without binding errors or inverted
limbs. They are sparse visual checks, not full contact-quality review.

Default remains v1.1: this small sample shows mixed contact/pose results and does
not justify a quality preference. Linux/macOS, CPU v1.2 inference, numerical parity
with upstream PyTorch, broad prompt coverage and isolated GPU benchmarks remain
unvalidated. The native installer/selection are functional; developer tooling
and all generated assets/weights remain outside runtime dependencies and Git.

Additional native-install end-to-end batch (same mutex): constrained v1.2
walk->wave exports 90 frames; combined keyframe/right-hand effector kick exports
75 frames. Keyframe pre-correction mean/max error 1.2/3.4 cm (corrected exactly),
effector mean/max 2.0/2.0 cm. This exercised the native-generated installed GGUF,
not a bypass of model verification. Native install's missing LICENSE/config
network path and both cached/native converter parity are validated; the full
weight transfer used curl before integration rather than being downloaded twice.

Integration: rebased onto origin/dev including native pose authoring, foot contact
cleanup and protected-pose retarget propagation; preserved all their source,
documentation and tests. Windows Vulkan Release rebuild passes all six current
CTests and the three development conversion tests. The v1.2 converter uses a
named immutable model record, so later additive model registries cannot change
its output verification by inserting entries. No dependency revision changed.


## Cleanup follow-up: mirrored constraints and reach tests (2026-10-28)

Gap review found that mirror omitted protected frame indices, despite retarget
and cleanup carrying them. Mirror now inherits constraint frames, previously
protectedFrames and pose-sequence start/arrivals using the same bounded source
recipe reader. It writes only the frame protection, not stale pre-mirror target
coordinates, handedness or constraint error claims. The protection means later
cleanup keeps the mirrored authored pose, rather than its original unmirrored
pose. No animation math, generation defaults or dependency revisions changed.
README describes protection through mirror too.

Corrected a validation gap: the old test labelled unreachable moved the root by
only 9 cm, leaving the target inside the unit-length leg's reachable region. It
therefore proved finite output, not unreachable behavior. The replacement moves
the root enough that the independent hip-to-anchor distance is >1.04 m and checks
ankle residual against the geometric minimum (distance minus the 1 m leg length,
tolerance 2e-5 m), while verifying both 0.5 m segments remain fixed. It asserts
the interval was solved, not skipped. Another test starts with exactly straight
knees, shortens the root-to-anchor distance and verifies ankle placement, stable
fallback bend direction and knee rotation steps below 0.05 rad/frame. These tests
pass without changing the solver; prior broad unreachable-validation wording
should be read subject to this correction.

Validation: Windows x64 ClangCL Release rebuild succeeds. Both affected CTests
(cleanup-contracts, cli-contracts) pass. CLI regressions exercise explicit/source
pose-arrival protection through standalone mirror, another batch mirror and batch
cleanup, checking all inherited indices. TEMP/TMP use the worktree-local
.research/test-temp-gap directory. Real end-to-end uses the existing 75-frame
parry-swing-return -> mirror -> cleanup (contacts both feet 0..15): independent GLB
FK verifies protected mirrored poses at 0/74 within 2e-6 per component, preserved
world ankle rotations and bone lengths (<9.3e-8 m); native checker reports zero
world-root/local-translation error and <3e-8 m bone-length change. Left contact
is restored by the drift-increased safeguard; right mean movement is 0.777 ->
0.679 mm/frame. Files stay ignored under .research/validation-gap. No inference,
GPU use or repeated visual preview was needed for this metadata/test-only change.

Useful remaining work: guard peak joint/ankle motion, including steps across the
contact interval boundary (mean drift alone does not rule out a pop); propagate
protected frames through connect/join with correct time offsets; broaden real
walking/turning/stair/pivot regressions; optionally solve authored hand/path
targets. Automatic contact suggestions, helper-node leg chains, sole/toe locking,
anatomical limits and ARM64 execution remain separate unimplemented/unvalidated
scope. The current direct-chain solver retains its documented opt-in controls.

Follow-up integration: rebased onto 3ef0599 (including concurrent DART research
and opt-in SOMA-RP v1.2 installation/generation), preserving their source, notices,
recipes and tests. Resolved append-only notes/CLI-test conflicts. Rebuilt the
integrated CPU executable and all six CTests pass. This does not repeat or claim
v1.2 model inference/conversion validation; the follow-up only changes mirror
metadata and native cleanup contract coverage.

## Pose placement and travel diagnostics (2026-10-28)

Investigated the reported backward-stumble recipe. A start and end pose at the
same world position intentionally asks world/auto placement to return home;
this is not a duration limit. The free mode masked root height/heading and body
while omitting root X/Z for later poses. This partial-root full-pose mask differs
from the complete-root conditions used by the pinned upstream multiprompt path.
The reproduced large drift occurs with a later floating pose; start-only free
and world have identical masks/values (now an explicit regression assertion).
Do not claim the placement switch alone changes a start-only clip.

Free poses now use two native Kimodo passes per seed. The guide samples the
same prompts, durations, seed, guidance and anchored poses/path, omitting the
floating poses. Its sampled hips X/Z locate those poses for a second complete-
root conditioned inference. Original floating semantics remain for error
measurement and correction (ground position is not snapped). No procedural
travel, blending generator, Python inference or hosted inference was added.
Cost: twice the sampling work when floating poses exist. Seeds in free mode
change behavior intentionally. Model/dependency revisions and licenses are
unchanged. Recipes include freePlacementPasses and inferredPoseLocations (world
coordinates before the optional final whole-clip shift). Auto selection of
world versus free, and world masks, remain unchanged.

`--end-pose idle.glb@0 --end-at X,Z[,YAW]` places the hips at absolute world X/Z,
keeps height, and optionally turns the entire pose to absolute hip-derived yaw
in degrees. Body-local rotations stay unchanged. Batch uses endAt: [X,Z] or
[X,Z,YAW]. Requires an end pose; in-place remains refused with any path.
A path sharing any pose frame first places/turns that pose before encoding and
correction, including frame zero's origin/heading. Explicit locations anchor
that pose even in free mode. Two poses or two path points at one frame remain
errors; a pose plus one path is now supported. Source files are not rewritten.
The full-body correction therefore lands the requested body at the new location,
without a separate connect step. Path-only points still receive no correction.

Variant sheet and GIF labels print horizontal displacement, accumulated path
length and peak per-frame horizontal root speed (metres, metres, metres/second).
Poses remain centred for comparison. The same world-space motion-root metric
is exposed by inspect and output recipes as rootTravel; recipe values describe
the final output, including correction and optional in-place removal. Custom
motion-root ancestors and scale contribute through FK. Labels were visually
inspected at 960x1036: all metrics and seeds are legible without overlap.

Validation: own Windows x64 ClangCL 18 Release Vulkan build, all five CTests
pass. New contracts cover merged pose/path position/yaw, preserved body-local
rotations, free-mode explicit anchoring, duplicate refusal, identical start-only
world/free conditioning, moving/stationary travel, CLI/batch endAt acceptance
and missing-end-pose refusal. An independent native exported-GLB check compares
all 30 joints at both ends against the sampled source and explicitly placed/
turned target (position squared error <1e-10, rotation <0.002 radians); passes.
Reproducible opt-in runner: tests/placement-inference.ps1, accepting user/self-
generated Idle, Models, optional Baseline and an engine-built Tide executable.
It holds TideMotionGpuValidation, runs native inference, checks correction and
pass counts, verifies exported endpoints, optionally captures a sheet and writes
travel.csv. No model or generated motion is committed.

Actual RTX 4080 SUPER Vulkan results: same existing self-generated idle, 2 s,
50 steps, guidance 2, seeds 1/3. Old / new displacement and peak speed:

| Prompt / seed | Old displacement m | New displacement m | Old peak m/s | New peak m/s |
| --- | ---: | ---: | ---: | ---: |
| stumble backwards / 1 | 4.403 | 1.038 | 3.932 | 1.114 |
| walk backwards / 1 | 4.170 | 2.613 | 3.967 | 2.269 |
| stumble backwards / 3 | 0.592 | 0.789 | 0.817 | 1.440 |
| walk backwards / 3 | 0.720 | 2.656 | 0.703 | 1.942 |

Explicit [0,-1,90] endpoint: 60 frames, 1.040 m displacement (the idle starts
slightly off origin), 1.190 m path, 1.803 m/s peak; endpoint verification passes.
Two further variants at [0,-1] produce 1.040 m displacement with peaks 1.646 and
1.312 m/s. Bounded three-frame TIDE_HIDDEN=1 contact-sheet capture succeeds,
using Tide built from D:/Projects/PurrEngine at
28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c and its matching staged native platform.
No engine changes. A guidance-4, no-correction free stumble also succeeds
(2.574 m, peak 2.674 m/s); an auto/free end-only walk succeeds (2.222 m, peak
1.557 m/s). These are functional samples, not a population quality guarantee.

Remaining gaps: the exact sword idle in the report was not attached, so its
16–19 degree mean / 52 degree maximum raw rotation error is not reproduced.
This work does not retrain the model or claim better raw rotation prediction.
Existing full-body endpoint correction makes rotations exact; modelError keeps
the uncorrected residual visible. Contact/velocity continuity can still be poor.
Textual "two steps" is still not a hard constraint. PROMPTING.md now explicitly
states that limit and describes authored pose-sequence targets for footfall
timing without claiming guaranteed intermediate contacts. No automatic step
count enforcement, IK, collision handling or research-only data is introduced.
Linux/macOS and broad prompt/seed quality sweeps were not run for this change.

Integration: rebased onto origin/dev 1033a9e, retaining concurrent native contact
cleanup, pose protection, DART research and optional SOMA v1.2 support. Resolved
append-only docs/tests and combined CLI field/option sets (model plus endAt).
The integrated Vulkan executable builds and all six CTests pass; the independent
exported placed-endpoint verifier also passes again. A requested repeat smoke
inference correctly refused because another task held TideMotionGpuValidation;
no conflicting inference started. Native measurements and sheet above came from
the successful earlier mutex-protected runs; the placement/inference algorithm
and platform.cpp are unchanged by integration. No v1.2 quality claim is made.

## Cleanup step guards and connected pose protection (2026-10-28)

Continuing the owner's gap-cleanup request. Native cleanup now checks individual
local joint rotation steps and world ankle 3D steps at every affected transition,
including start-1 -> start and end -> end+1 when present. Comparing only average
horizontal drift or a whole-interval maximum can hide a release pop; checks use
the original value at the same joint/transition. Quaternion angles use shortest
arc atan2 to avoid acos precision near identity and quaternion-sign false pops.

New optional contact-plan controls (also used by batch's existing contacts file):
maxJointCorrectionDegrees 0..180, default 30; maxJointStepIncreaseDegrees 0..180,
default 5; maxAnkleStepIncrease 0..0.5 world metres/frame, default 0.005. Numeric
slack is 0.001 degree / 1e-6 m. These are conservative introduced-change bounds,
not anatomical limits or a speed limit on existing kicks. Zero allows only the
numeric slack; higher values deliberately permit larger edits. All settings are
finite, type/range checked in complete batch preflight before any inference.
Generation/retarget defaults and inference/model arithmetic remain unchanged.
The additional defaults affect only the already opt-in cleanup command. The
unit fixtures and actual rejected Synty release step below justify adding them.

Any violated guard restores the entire original interval. Existing correction-
bound / drift-increased checks remain; new skip reasons are joint-correction-bound,
joint-step-increased and ankle-step-increased, tested in that order after drift.
The recipe records effective settings, safeguards=tide-contact-bounds-v1, final
peak joint/ankle steps before/after and attempted max correction/step increases.
Attempted values survive rollback; final values describe the exported asset.
Boundary steps are recomputed after all decisions so adjacent intervals cannot
leave stale metrics. Mean drift/residual keep their existing inclusive-interval
meaning. Independent leg chains are required: custom mappings that put one leg
under the other are now refused transactionally, since their corrections would
interact despite distinct joint indices. This is a mapping validation safeguard,
not automatic bone identification. Existing normal SOMA/mapped rigs still work.

connect recipes now always protect bridge endpoints. With --joined they also
carry all inherited authored source/target indices using exactly join()'s duplicate
endpoint omission: bridge begins at sourceFrames-1; target frame zero is at
sourceFrames+bridgeFrames-2. Target pin zero binds to the retained bridge endpoint.
Bridge-only exports retain just their two endpoints because input interiors are
absent. Inherited recipe validation occurs before an inference model loads.
Target --offset changes the retained poses but not their frame indices. As with
mirror/retarget, protection refers to the resulting exported pose and does not
copy stale constraint coordinates/error claims. A model-free core helper shares
the frame-map contract with independent count/pose tests. No join/inference
animation math changed; pins do not promise smooth velocity through the joins.

Windows x64 ClangCL 18.1.8 CPU Release builds with project warnings as errors.
All six CTests pass, with TEMP/TMP in .research/test-temp-bounds. New independent
contracts: a 21 mm release step after an interior lock (input 1 mm/frame) is
rejected and restored; per-joint release pops and absolute correction bounds;
shortest quaternion-sign arcs; final metrics across adjacent reverse-ordered
intervals; refusal of interacting serial leg chains; exact pin mapping against
an independently assembled 8+26+8 frame join; bridge-only/minimal maps and bad
indices/counts; CLI/batch guard range/preflight refusal before model loading.
The initial synthetic join fixture used a bridge targeting input frame 4 with
a target starting at frame 0; its new pose assertion correctly failed. The
fixture now supplies actual matching endpoints; production join code was not
changed to accommodate a contradictory test.

Real asset checks reuse the earlier self-generated idle and the scratch Synty
character (neither committed). Default guarded idle cleanup retains the earlier
mean horizontal reductions: left/right 0.273/0.261 -> 0.062/0.153 mm/frame. Final
peak joint steps 2.290/0.893 degrees, full 3D ankle steps 2.204/4.242 mm/frame.
Those peaks can rise within the configured per-transition allowance; this does
not claim every derivative improves. Synty left mean 0.301 -> 0.078 mm/frame;
right is now restored to 0.332 mm/frame: its attempted extra ankle step is
6.145 mm/frame, beyond the 5 mm default, despite improved mean drift. This is
an intentional safety/cleanup tradeoff. Independent tests/verify_cleanup.py now
checks per-transition bounds and final reported peaks from exported GLBs, alongside
root/local-translation, protected pose, world ankle rotation and length checks.

Real native end-to-end reproduction: tests/cleanup-inference.ps1 -Models DIR
-Source GENERATED_SOMA.glb -Target GENERATED_SOMA.glb [-Tool EXE]. Developer Python
stdlib is used only for independent exported-asset verification (verify_connect.py
and verify_cleanup.py); no Python is used by inference, build, installation or
CTests, and tests are not packaged. The script creates a GUID work/temp directory,
adds an interior protected frame to each input recipe, runs joined connect,
bridge-only connect and cleanup in one ordered batch, and checks one native ORT
session load. It uses MotionBricks's CPU-only backend with one CPU thread and
records no performance timings. Initial attempts to reserve TideMotionGpuValidation
found it busy and started no inference; the final functional test runs only CPU,
with no GPU, VRAM allocation or GPU benchmarking. The GPU mutex remains mandatory
for GPU inference/benchmarks; it was not bypassed for any GPU work here.

Actual source/target: existing parry-swing-return.glb, each 75 frames, 8-token
bridges, target offset [0,0,0.4]. Output joined=174, bridge=26; source/target
inherited pins plus seams are [0,1,74,99,100,173], bridge pins [0,25]. Independent
GLB comparison checks every retained source/target pose (translation <2e-6 m,
rotation <0.002 degrees), including the offset target, and all metadata indices.
Cleanup's explicit two foot contacts 0..15 are both restored by drift-increased
in the presence of the extra authored frame-1 pin. All six protected poses remain
within 2e-6 per component. Independent max length error <5.2e-8 m; native checker
world-root/local-translation error zero, length error <3e-8 m. These conservative
skips are reported, not presented as successful foot locking. Accepted real
cleanup and its independent metrics are the idle examples above. Actual files/logs
remain ignored in .research/cleanup-connect-f3602d03b6de49babeb6b9dd2b8c06ef and
.research/validation-bounds. No evaluation motion, weights or character files
enter Git. No repeated visual capture was needed for pin metadata and stricter
rollback of the previously inspected poses; no new temporal-quality claim.

Remaining scope: numerical safeguards are not a collision/anatomical-limit or
acceleration/jerk solver; no automatic contact suggestions, sole/toe locking,
helper-leg chains, hand/path IK or guarantee of gait/flight semantics. ARM64,
Linux/macOS and broad walking/turning/stair/pivot sweeps remain unvalidated here.
Model/dependency pins and all upstream notices are unchanged.

Integration completion: rebased onto origin/dev 670b6b6, preserving concurrent
MoMask source/representation/ORT proof work, its notices and packaging notes.
The integrated Windows CPU Release rebuild succeeds; all seven CTests pass,
including the new incoming HumanML contract test. No MoMask network inference or
quality claim is made. No inference algorithms/dependency revisions changed in
this rebase, so the successful native connect/cleanup asset checks above remain
applicable without repeating model inference.

## Broader paired v1.1/v1.2 quality comparison (2026-10-28)

Owner requested more evidence for choosing a default. Full method, thresholds,
results, uncertainty and limits: notes/soma-rp-comparison.md. Added developer-only
400-clip protocol/runner, independent exported-GLB geometry scorer, three model-
free evaluator checks and optional EXCLUDE_FROM_ALL/uninstalled soma-sheet helper.
No production inference, dependency/model pins, license terms or defaults change.

Native Vulkan generation at 15c78fd: 400/400 clips completed under the shared
TideMotionGpuValidation mutex, four counterbalanced blocks, fresh seeds 101..108,
ten plain prompts at 2.5/5 seconds plus common-pose kicks, curved paths and two
multi-prompt sequences. All expected frame counts, finite FK and model identities
pass. Geometry proxy passes 164/192 v1.1 versus 167/192 v1.2; paired seed-cluster
bootstrap interval for the difference -2.60..+5.73 percentage points. Not an
overall winner. Plain 5-second right kicks improve 1/8->6/8 (short kicks both
8/8); backward/turn foot slide worsens while crouch slide and path error improve.
Raw pose-kick position error slightly favors v1.1 with a v1.1 idle reference.
Twelve preselected randomized A/B pose-sheet votes: v1.1 2, v1.2 3, seven ties.
Sparse visual review is not a full-motion/human-panel study. v1.1 remains default;
use case-specific selection rather than claiming higher version implies quality.

Every block observed another tide-motion process (workload not inferred), so
performance is not isolated. The GPU mutex was released for the concurrent ARDY
measurement and reacquired for twelve bounded hidden captures with checkout-built
Tide. Optional GIF correctly refused: that engine checkout lacks --record; no
animated validation claimed. Weights, generated clips, raw measurements, logs,
executable/reference hashes and hidden review keys remain ignored .research data.
The new scorer's three checks and current six CTests pass; soma-sheet builds and
renders all twelve two-view PNGs. No extra CI dispatch or PR is requested.

Comparison integration: rebased onto origin/dev 93a92e3, preserving the full
native ARDY backend, its validation/notices/packaging, incoming HumanML proof
work and cleanup guards. Windows Vulkan Release rebuild passes all eight current
CTests; the optional soma-sheet target builds and the evaluator's three model-
free tests pass. Runtime staging includes the new comparison note and excludes
the developer-only renderer/tools. The 400-clip measurements retain their exact
15c78fd generation provenance; no unchanged Kimodo inference was repeated for
this additive test/documentation integration.
