# Native animation authoring

## Scope agreed with the owner

Separate repository, CLI and installer. Native executables with minimal user
dependencies; no required Python environment. Tide's CLI and engine remain
independent. Kimodo creates motions; MotionBricks creates clip-to-clip transitions.
Ordinary exported animation assets are the game interface. Tide can render previews.

## Native contracts

The core is C++23. Kimodo is compiled as a static library from localai-org/kimodo.cpp
at ed8a931aedb132f768190f6931c365855caba74c, including its GGML submodule at
8c63e70982c95ceb862e3a1073a2c1beef75d60a. This port supports the older SOMA RP
v1.1 checkpoint. It does not claim support for the newer v1.2 checkpoint.
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
The native dependency's constrained/multi-prompt APIs are not exposed yet.

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
