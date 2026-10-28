# Native animation authoring

## DART complete authoring controls and native optimization (2026-10-28)

Supersedes the earlier three-label/source/gradient capability limits below.
Owner asked to expose the model's useful capabilities and continue implementation.
Delivered native free-text timelines; saved history or explicit SOMA/mapped humanoid
source conditioning; single-pose repeat; actor sample/mean/none; per-segment CFG;
DDIM eta/ancestral DDPM; original-schedule timestep subsets; initial latent scale/
zero initialization; ground height and opt-in floor projection. No Python runtime,
new user library, hosted service, procedural generation or pose blending. Existing
Kimodo/ARDY/MoMask defaults and code paths remain independent. Current schema,
examples, exact limitations and reproduction commands are in [dart-controls.md](dart-controls.md).

Native CLIP BPE reads one additional exact MIT archive (1356917 B,
924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a), pinned to OpenAI
CLIP d05afc436d78f1c48dc0dbf8e5980a9d471f35f6. Existing ufbx inflate avoids a new
library. Generated Unicode 14/CPython 3.11 fact tables, licenses and generator are
included; no vocabulary/learned data is embedded. Sixteen official-tokenizer
comparisons cover English, contractions, case, accented/decomposed French, Greek,
Chinese, Korean, emoji, entities, empty text and truncation. No arbitrary prompt
whitelist remains. ftfy mojibake repair and some malformed numeric entity mappings
are not implemented; the normal valid UTF-8 path is tested. Embeddings cache up to
4096 prompts; motion and gradient sessions persist across batch actions.

The prior exact-rights audit is retained: the author's public Apache-2.0 checkpoint
statement, pinned release files and released normalizer support this inference set.
AMASS/BABEL/HumanML training access is separate; no body model, SMPL rest extract,
stand.pkl, upstream evaluation history or dataset is downloaded/used. Body-free
fixed geometry comes from this model's generated positions/rotations, and pelvis
offset from released inference coefficients. Native source adapters use explicit
known topology/mapping and existing retargeting, not automatic bone identification.
No person was contacted. No model/ONNX/tensor/reference/generated motion enters Git
or runtime packages. Existing denoiser/VAE/actor/CLIP/tensor pins remain unchanged.

Implemented real analytic native optimization: six additional standard-opset-17
forward/VJP schemas (decoder gradient 1742 nodes; denoiser gradient 2618; history/
world forward 546 and gradient 1618; SOMA FK 2364 and gradient 4474). Full operator
inventories, sizes/hashes, shapes, primary URLs and licenses are in dart-models.json.
All learned tensors reuse the same verified external 372168788 B file. Export uses
pinned development-only PyTorch 2.7.1+cpu source e2d141dbde55c2a4370fac5165b0561b6af4798b,
math SDPA/ATen decompositions, torch.func VJP and fixed symbolic lowering of softmax
backward/slice_scatter. PyTorch BSD notices are retained; no training runtime ships.
All inline floating constants were audited as mathematical scalars/zero masks;
learned initializers remain external. Graph declarations moved to one cold-build
translation unit; per-node cold lambdas avoid the giant-function stack overflow
encountered during native schema construction. Native preparation matches all pins.

Native Adam optimizes reparameterized latent noise through every CFG denoiser,
DDIM/DDPM step, decoder, history recanonicalization and world transform across the
entire timeline. Losses cover endpoint poses, arbitrary frame/joint/axis constraints,
segment goals, max jerk, relative initial foot floor and owner-supplied dense SDF
joint penetration. SDF samples are trilinear, border-clamped with zero outward
derivative. The released actor is not a latent-optimization prior. Best measured
candidate is retained; seed poses remain fixed, endpoints are soft. No snapping,
blending, IK or finite-difference optimization. Optional directional finite
differences only verify the analytic gradient. Iteration progress is printed.
Default objective is canonical 30-joint SOMA FK including requested floor projection;
raw 22-joint prediction objective remains selectable. Each default optimized asset
independently checks graph coordinates against native retarget/FK, fails above
0.2 mm. No SMPL body/surface/shape objective is claimed; fingers/eyes/jaw have no
independent DART degrees of freedom. Scene joint penalties are not mesh contact,
self-collision or collision-free motion guarantees.

Source GLB/FBX animation names, selected frames, explicit maps and native first/last
history states preflight before inference; earlier batch outputs are accepted as
inputs and split execution groups. Existing-output protection is preserved. Native
history saves both unnormalized rows, world transform, fixed reference and normalizer
identity. Recipe inputs have SHA-256/size/path provenance. Batch forward inference
still groups up to eight independent streams; optimization actions reuse sessions
but run separately. Chunks/reverse history remain sequential, not falsely batched.
Output is two initial frames plus eight/chunk, including repeat; source prefixes
are not copied. Budgets remain <18000 frames. All model files stay outside Git.

Windows ClangCL 18.1.8 Release/RTX 4080 SUPER, four CPU threads:

- CPU/GPU six-graph reference comparison passes elementwise 2e-4 absolute + 2e-4
  relative bounds. Maximum large denoiser gradient error .000249386/.000256062;
  decoder VJP <=1.526e-5, transforms <=7.63e-6. Larger absolute derivative errors
  are covered by the relative bound, not hidden by reporting only forward errors.
- Whole-chain four-direction numerical checks pass on multi-chunk SOMA and raw
  joints, DDPM, stochastic respaced DDIM, floor, jerk, SDF and sparse constraints.
  Largest observed directional error in the eleven-asset test <9e-6 at epsilon .05.
- tests/dart-controls.ps1 generates eleven real own-prompt assets with one model/
  session set: wave/squat/jump, native/source/repeat continuation, actor mean and
  an untrained backward-walk prompt, full-pose targets and scene regularizers.
  Two six-chunk/50-frame inbetweens, 120 iterations/rate .05: objectives
  .02161793/.00806799 -> .00000287268/.00000325300; independent max endpoint errors
  over all 30 joints .00859128/.00964742 m. 30.55/30.02 s. These reachable target
  poses were generated with the same model, not selected from evaluation data.
- Independent test dart_controls_metrics.py confirms native-history/source first
  two poses to 2.84e-7/3.42e-7 m and fixed seed poses. Independent GLB/FK lengths
  <2.5e-7 m; 723 native half-frame samples position <2.4e-7 m/quaternion dot <7.3e-8.
  Exported constraint errors/masks are independently recomputed in dart_metrics.py.
- Own free-text adherence proxies: wave hand movement relative to pelvis spans
  .458/.966 m; squat pelvis height .205.. .859 m; jump both-feet minimum peaks .206 m.
  These are geometry observations, not a semantic benchmark. Segment-boundary root
  steps .00215/.00418 m, velocity changes .114/.126 m/s in that timeline. Largest
  joint step 25.39 degrees. Endpose foot-slide proxies 3.52/3.55 cm/frame; contact
  quality is not fixed. A too-short 18-frame goal probe remains .662 m away.
- Eleven-action sampled host peak 1124900864 B, GPU peak 680607744 B; sessions reused.
  A later twelve-action/four-seed regression reaches 24/24 locomotion goals in
  3.532 s incl verification/load, sampled host 773713920 B, default GPU batch 8.
  Older matched batch-1/8 CPU/GPU timings below remain historical observations.
- Additional packaged tests cover native targetFrame 0, mapped source/target,
  named animation selection, repeat, stochastic sampler, analytic gradients and
  SHA input provenance. Assets reimport and independent exported error checks pass.
- All twelve Windows CTests pass; Ubuntu GCC 13 warnings-as-errors build/twelve
  CTests, native six-graph parity, tokenizer and actual CPU optimized GLB pass.
  Isolated per-worktree temps prevent cross-thread collisions. No macOS GPU or
  full upstream SMPL rollout equivalence is claimed.
- Packaged executable runs GPU by default, strict GPU refuses missing runtime,
  auto actually generates on CPU with Windows-only PATH. ZIP installer/reinstaller,
  path traversal and model-preservation checks pass. New controls/docs/licenses
  are installed. The owner's installation is untouched.
- Hidden 960x540 native mannequin capture at --frames 51 inspected successfully,
  using Tide CLI/compiler/native platform built from clean D:/Projects/PurrEngine
  367ae4da2d31913176ae1e2624e1d78a1204f03c. Private staging needed both release and
  mingw-release runtime files; full engine install stops at unrelated unbuilt
  tide-editor. No engine sources changed and no engine installer claim is made.
  TideMotionGpuValidation held for every GPU inference and renderer test.

DART's distinct value versus independently implemented ARDY is eight-frame action
changes, released waypoint actor and reverse full-timeline latent optimization.
No ARDY implementation was duplicated or quality-ranked. Remaining limits are
measured prompt/contact/constraint quality, fixed representation/checkpoint,
sequential optimization requests, no SMPL shape/mesh workflow, no independent fine
hand animation, and tokenizer repair differences. No training, research assets,
automatic rigging/skinning or non-humanoid claim.

First combined integration rebased onto c239748, preserving the concurrent ARDY, MoMask,
first-person/equipment and native advisory review work. CMake's source-list and
notes insertion conflicts retain both sides. GCC 13 exposed five incoming
misleading-indentation warnings (GLB/FBX bounds and review radius validation);
only statement layout was changed, with no behavior or inference arithmetic change.
All fifteen combined Windows and Ubuntu CTests pass; affected native review,
preview, CLI, retarget and DART checks pass after those formatting fixes.
Six new gradient/mathematical graph profiles show only DmlExecutionProvider:
3/3/1/80/1/120 kernel events for decoder-VJP, denoiser-VJP, transition,
transition-VJP, SOMA FK and SOMA VJP respectively; zero CPU-provider kernels.
These graph partitions are not a claim that host control/text encoding runs on GPU.
A fresh Windows native cache reuses the five hash-verified raw model files,
fetches BPE by the pinned HTTPS URL and constructs/validates thirteen schemas and
shared tensors without Python. All sixteen tokenizer fixtures pass on that cache.
Final Windows ZIP installer/reinstaller and installed GPU/default/strict refusal/
actual CPU fallback tests pass; installed native GPU gradient optimization also
passes with Windows-only PATH. The native Linux archive/installer passes, and its
installed CPU binary generates/validates real goal and optimized GLBs with system-
only PATH and no LD_LIBRARY_PATH. A reused scratch output initially caused the
expected overwrite refusal; the subsequent native run uses a fresh unique directory.
The combined preview also captures the second optimized asset at --frames 51
using the same checkout-built engine; image inspected. No owner installation,
engine source, model data distribution, force push, extra CI dispatch or PR.

Final rebase onto f08cb0e also preserves native equipment/grip/object/bow authoring.
Twenty further GCC misleading-indentation warnings in incoming equipment source,
export and CLI are fixed by splitting statements only; whitespace-stripped source
is identical to origin/dev. All eighteen combined Windows/Ubuntu CTests pass;
affected equipment, CLI and DART checks pass after the layout fixes. DART model
arithmetic, schemas and previously measured assets are unchanged. Native optimizer
and forward GPU profiles remain valid; no expensive inference benchmark is repeated
for these unrelated source-layout fixes. Final package and installed runtime smoke
checks cover the combined executable before the direct non-force dev push.

## Native equipment grips, object export and bow release (2026-10-28)

Owner explicitly requested implementing the remaining tool authoring work and
allowed parallel model investigation. Native `equipment inspect|bake|preview`
now operates on existing body GLB/FBX clips with a strict version-1 JSON plan.
See [equipment-authoring.md](equipment-authoring.md) and examples/equipment for
commands, complete schema and five-frame sword/shield, two-hand and bow templates.

Object segments cover every source frame without gaps/overlaps. They name world,
explicit body bone or another object as their anchor, with authored position,
Euler rotation and mesh-only scale keys. Object hierarchy is evaluated in metres
with scale stripped from anchor frames; each resulting object becomes an independent
world-space GLB root track. Object trajectories are sampled from the original body
before optional arm correction, avoiding recursive hand/object chasing. Segment
seam translation/rotation steps are reported. Key/segment bounds, cycles, names,
finite values and source frame coverage are checked before sampling.

Grips measure every authored contact frame, including an optional palm offset,
position/orientation residuals and object-relative drift. Explicit direct three-bone
arm chains can receive analytic offline IK via `--correct-grips`. Body root and
all translations, fingers, outside intervals and protected frames are preserved.
Source recipe protected frames are carried in automatically. Excessive correction,
rotation or boundary-step growth, unreachable targets and protected-frame residuals
reject/restore the whole affected interval. Interacting/overlapping chains refuse.
Final metrics use the actual accepted body; no failed attempt is reported as fixed.
`--strict` refuses unsatisfied sampled grips before writing outputs or previewing.

Bow limbs are ordinary authored object parts. Two string rods follow independent
top/bottom/nock object frames, so articulated limb tips and a hand/nock target can
share exact endpoints. Nock keys author draw/recoil. A `release` event must name an
object at an attached-to-world segment transition; subsequent arrow flight is an
explicit world track, not projectile simulation. Events retain frame/time/payload
in GLB extras and the recipe. Ordinary channels interpolate LINEAR between baked
samples; this is a sampled authoring/contact contract, not subframe physics.

`bake` embeds rigid equipment GLB meshes with static scene hierarchy, attributes,
core materials, embedded PNG/JPEG textures and source asset attribution. Supported
material extensions are explicit. Unsupported animation/skins/morphs/cameras or
external resources refuse. Untextured rigid FBX converts through existing pinned
ufbx in Y-up metres, with appearance losses stated in meshConversions; textured
FBX refuses in favor of self-contained GLB. The body skeleton animation is included;
the character mesh is not. Optional `--body-output` writes a reusable corrected
body-only clip/recipe. Native batch `action: equipment` supports this final authoring
stage, with output-path collision checks including optional bodyOutput.

Exports and JSON files use exclusive partial creation and no-replace same-directory
hard-link publication. This fails safely on filesystems without hard-link support;
multiple output files are not a single filesystem transaction. Recipes record the
effective plan, source/mesh/map/character hashes as applicable, selected source clip,
tool version, protected frames, decisions and errors. No inference backend/model,
dependency revision, engine source or owner installation is changed.

Windows x64 ClangCL 18.1.8 Release passes all sixteen CTests. New contracts cover
independent FK for two moving hand targets, centimetre rigs, palm offsets with
preserved hand orientation, rollback/bounds/protected/outside/root preservation,
cycles and invalid release timing, string endpoints and independent arrow flight.
Final GLB tests parse actual embedded bytes with cgltf: body/object/scale tracks,
quaternion continuity, repeated textures/material remapping, PNG bytes, attribution,
nested static transforms, rod winding, FBX metre conversion and overwrite/failure
cleanup. CLI covers strict no-output refusal, body-only reuse, recipes, batch paths
and execution without model-cache environment discovery.

Seven/eight-frame own camera fixtures and own box prop geometry exercise real
authoring previews with `--frames 9/15`, first-person capture and eight-frame GIF.
The corrected synthetic draw interval passes strict validation: max marker position
error 1.2287812e-7 m. Initial unreachable fixture was correctly rejected; its authored
nock target was moved into reach. These numbers validate IK/track plumbing, not
model weapon quality or finger contact. Standalone exported GLB also decodes with
the current engine and plays through its ordinary SkinnedMesh animation path.
The initial standalone test camera framed the wrong X side: Tide reflects glTF X
on import; correcting that camera verifies the original independent-root export.
An unnecessary trial rigid-binding workaround was removed. Captures were inspected.
All renderer runs held TideMotionGpuValidation, TIDE_HIDDEN=1 and explicit bounds,
using the same rebuilt/staged engine checkout 367ae4da2d31913176ae1e2624e1d78a1204f03c.

Parallel [Dex2HOI release audit](dex2hoi-release-audit.md) recovered the precise
public generation checkpoint/statistics listing and executed a bounded native
18D host-decoder probe on 256 synthetic cases. Maximum translation error 6.91e-7 m,
rotation-matrix error 9.16e-7 vs independent double reference. No learned network,
checkpoint/statistics payload or evaluation motion was used. Missing args.json
means upstream default fallback, not automatic refusal. Actual checkpoint/code
grants and exact trained configuration remain unresolved; no backend is integrated.

Limits remain explicit: no collision/penetration or anatomical certification,
automatic finger grasping, inferred weapon semantics, automatic body skinning or
non-humanoid retargeting. These authoring controls supplement real learned body
generation; they do not replace it with procedural generation or pose blending.

Integrated additively onto origin/dev c239748's advisory review/calibration work.
Preserved both CLI namespaces, review source/tests and implementation notes.
The combined Windows Release build and all eighteen registered CTests pass;
equipment arithmetic and exported assets are unchanged by the integration.

## Native advisory review and human calibration (2026-10-28)

Owner requested an agent-facing observation environment with human judgment,
situational exceptions and feedback that can be associated with motion locations.
`review MANIFEST.json -o REVIEW.html` now builds a self-contained offline skeleton
review and adjacent agent JSON. Contracts, limits and feedback semantics are in
[advisory-review.md](advisory-review.md). No model/dependency/pin/license changes.

Native observations localize shortest quaternion steps, foot centres below an
explicit reference plane, horizontal movement of low foot/toe samples, forearm /
torso capsule overlap and explicit bone-local prop / torso capsule overlap.
Intervals, subjects, peak frames/values, thresholds, units and approximation methods
remain distinct from human interpretation. No quality score, severity or pass/fail;
no observations does not imply successful validation. Custom unmapped rigs get
rotation observations with canonical body checks explicitly unavailable. SOMA-77
is reduced for body-only review; fingers are not evaluated. Input clips never change.

The dependency-free page provides front/side playback, scrubbing, speed controls,
optional capsule display, selected-interval loops and hidden/revealed observations.
Humans can record accurate/actionable, accurate/acceptable, intentional, inaccurate
or unsure assessments, plus timestamped missed issues and reasons. Assessments are
counted separately, without an overall quality score. JSON export/import retains
measurements and human feedback separately. Source/mapping SHA-256, animation,
settings and rule version identify the review; mismatched imports are refused.
Feedback is held in the open page and must be exported before closing. Nearby
observation associations are suggestions, not a causal diagnosis. JSON script-data
escaping and text-only labels/notes prevent source/feedback markup execution.

Windows x64 ClangCL 18.1.8 Release CPU build passes all 15 CTests. New native checks
use independent 90-degree/sign-invariant rotation oracles, split/merged intervals,
degenerate/separated capsule cases, unavailable custom-rig coverage and malformed
settings/prop refusals. CLI checks exercise source-relative manifests, observation
sidecar, overwrite refusal and a late missing source leaving no outputs. Existing
13 CTests remain passing. Developer-only existing headless Chromium checks pass
hidden/revealed hints, selection, intentional assessment, notes/nearby association,
export/import, mismatched identity/invalid timestamp refusal and scrubbing/stepping.
No browser or Python dependency is added to the installed executable.

`tools/create-review-set.ps1` reproduces a private eight-case review, optionally
nine with an existing generated clip, using the owner's sword-combat library.
The uninstalled `review-examples` helper extracts a real guide pose and creates
explicit diagnostic mutations (root -.15 m, 70-degree wrist pulse) and a deliberately
misaligned prop proxy; these are labeled test edits, never model outputs. Actual
native MotionBricks generates two held eight-token transitions from library idle
to extracted parry and back, yielding 51 frames. The assembled local nine-case set
includes unmodified-source idle/parry/roll through explicit SOMA retargeting, an
existing own-prompt Kimodo cut, that fresh guided inference and diagnostic cases.
Original source files/mappings are hashed, and all licensed/derived assets, poses,
recipes and review outputs remain ignored under output/. No evaluation motion or
research-only checkpoint is used or distributed. Script reruns refuse existing
directories, preserving any human feedback. No additional CI dispatch or PR.

Retargeting reports discarded non-root translations on these purchased FBXs.
Therefore the review expressly describes canonical proxy motion, not original
skin accuracy or original exact poses. All analysis is on sampled frames; there is
no calibrated anatomical reachability/joint-limit test, mesh or continuous collision,
balance/force certification, aesthetic judgment, automatic repair, feedback-trained
model or direct Tide pose-editor integration. Screenshots/UI were inspected as
review-interface checks. No engine renderer or GPU work ran; TIDE_HIDDEN=1 was set
for native invocations. The engine checkout and owner installation are untouched.


## Dex2HOI audit and native first-person/equipment review (2026-10-28)

Owner requested further Dex2HOI investigation and fixing encountered Tide limits.
The [pinned audit](dex2hoi-feasibility.md) records source revision
a92aca6b48b7a0c9a45a1465aa984213d3c8ceda, real inputs, native operator work,
representation/normalization and release gaps. The released HUMOTO demo uses
dataset-backed histories/state/geometry and is not an arbitrary text + meshes CLI.
Its PointNet geometry cache, CLIP, dual-stream/fusion networks and forward DDIM
path suggest a native route, but no export, inference/parity or speed claim is
made. Top-level code/checkpoint grants and exact checkpoint inventory remain
unverified. Source was inspected in ignored scratch; no upstream implementation,
checkpoint, body model or evaluation motion was imported into Tide or Git.

Implemented tool fixes independently of model integration:

- `preview --first-person` provides explicit bone or root-following eye cameras,
  local metre/Euler offsets, vertical FOV and near plane. Root mode carries the
  first-frame eye transform on the animated motion root. Rendering uses Tide's
  existing matrix camera API; no engine source change or dependency update.
- First-person mannequin renders only arm geometry while retaining all 30 SOMA
  joints, inverse binds and original animation. Hidden capture initially exposed
  head/nose geometry crossing the root-following view during a nod; arm-only
  presentation fixes that occlusion. Custom rigs use supplied skinned arms meshes;
  no automatic mesh cutting/skinning, arms-only inference or export is claimed.
- Version 1 `--setup` / `--save-setup` persist props and camera, validate unknown
  keys/ranges and refuse overwrite. Loaded mesh paths resolve relative to the
  JSON location; saved paths are absolute. Character/map/texture selection is
  intentionally explicit at replay. CLI camera/prop options cannot override a
  loaded setup silently. Setups are review configuration, not model provenance.
- Third-person framing reads actual GLB/FBX geometry with existing pinned cgltf /
  ufbx, including active GLB scene/nested transforms and FBX axis/unit conversion.
  Conservative socket-centred spheres include mesh bounds, offsets and every
  authored frame. Uniform inherited rig scale affects socket positions but does
  not shrink held meshes/offsets. Explicit unweighted socket nodes and ancestors
  join the sampled rig for bounds/calibration. This framing can leave generous
  space; it is not continuous collision detection. First-person FOV stays authored.

Windows x64 ClangCL 18.1.8 Release CPU build passes all thirteen CTests. New native
contracts cover active-scene/nested mesh bounds, odd-frame extremes, scaled rigs,
unweighted sockets, Euler order, setup paths/roundtrips/refusals, and arms geometry
with its complete rig. FBX metre bounds and CLI camera/setup preflight are covered.
Earlier prop CLI fixtures were skeleton-only files; they now use self-authored
triangle meshes because real bounds correctly reject assets with no mesh positions.

TIDE_HIDDEN=1 engine captures at seven frames validate both camera modes, setup
replay, a 4.2 m test blade in third person and mapped separate skinned arms. A
15-frame custom-arms GIF run records eight frames. Images were visually inspected:
arms/equipment remain visible without the mannequin head intrusion; the long blade
fits in both panels. These are synthetic camera/attachment fixtures, not generated
weapon skill or grip-quality evidence. Engine CLI/native platform remain from
checkout 367ae4da2d31913176ae1e2624e1d78a1204f03c, rebuilt/staged as documented
below; TideMotionGpuValidation held for renderer runs. Fixtures and captures remain
ignored. No owner installation, model arithmetic, retarget algorithm or pin changes.

Remaining substantial authoring features are explicit: continuous grip diagnostics
and correction, model object-space contact inputs, independent object-track/mesh
GLB export, and bow articulation/release events. They are not solved by a camera
or rigid preview attachment. Dex2HOI has not been integrated or quality-ranked.

## Props and first-person research / multiple attachments (2026-10-28)

Audited dev 8d535b7 and primary model papers/release repositories. Findings,
source links, candidate release status, model/tool distinctions, first-person
workflow and proposed acceptance measurements are in [props-first-person.md](props-first-person.md).
Dex2HOI is a released two-object lead; Text2HOI a hands/object lead. Neither was
run or established as a weapon solution. No new model, dependency, body asset or
evaluation motion was downloaded, integrated or distributed; existing pins and
licenses are unchanged. Full-body inference plus camera-aware review is proposed,
not a validated arms-only generation/export feature.

Implemented repeated preview `--prop MESH --prop-bone BONE` groups, each with
independent bone-local offset and Euler rotation. A new group resets transforms
and requires its own bone. Existing single-prop usage remains supported, including
options before the first prop. Mesh copies and generated scene identifiers are
indexed, so repeated input paths/names cannot overwrite each other's staging.
All bones are validated on the drawn rig. Cached prop assets are removed when
the attachment count decreases, including legacy single-prop filenames. CLI help
and README document grouping and examples.

Windows x64 ClangCL 18.1.8 Release CPU build and all twelve CTests pass, including
multi-prop GLB/FBX acceptance through scene creation and missing/invalid second
bone or mesh refusals. Independent generated-scene checks verify both bone
bindings and distinct transforms. Hidden seven-frame captures with two, one and
zero attachments validate compilation, rendering, transform reset and stale-asset
cleanup. Two-prop front/profile capture visually inspected using self-authored
box blade/shield test geometry and an existing synthetic rig fixture; this tests
attachment plumbing, not sword action or grip quality. Captures/assets stay ignored.

Renderer validation uses Tide CLI and native platform rebuilt from engine checkout
367ae4da2d31913176ae1e2624e1d78a1204f03c at D:/Projects/PurrEngine, privately staged,
with TIDE_HIDDEN=1, --frames 7 and TideMotionGpuValidation held. Initial staging
attempts lacked native toolchain files; runtime components were then staged from
the same checkout's mingw-release build. Full installer stopped at an unrelated
unbuilt tide-editor target; this is runtime preview validation, not installer QA.
No engine source or owner installation changes.

Remaining gaps: preview attachments do not condition inference or become GLB object
tracks; no secondary-hand grip solver, collision/contact guarantee, bow articulation
or release events. Framing still ignores prop bounds. First-person camera review,
arms mesh/export preparation, equipment recipes and continuous grip diagnostics
remain explicit next work; SOMA-77 expansion does not infer finger grips.

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

## Owner-authorized native MoMask local experiment (2026-10-28)

The owner explicitly accepted unresolved checkpoint scope for local use after the
Hugging Face audit ("just use it and deal with it later"). This supersedes the
earlier stop-at-license gate for this local experiment, not the factual rights
findings. Commercial clearance is not asserted; weights remain outside Git and
releases. Current implementation, commands, validation and gaps are documented
in [momask-local.md](momask-local.md). Source/configuration/CLIP revisions remain
pinned; full private input and native bundle hashes are in momask-private-*.json.

MOTION_MOMASK_EXPERIMENTAL defaults OFF. Its separate tide-momask native launcher
and optional tide-motion momask namespace add text generation, temporal inpainting,
native local model import, complete preflight, recipes and ordered session-reusing
batches. It uses existing C++ ORT 1.31.1 CPU, native ASCII CLIP BPE/sampling/RNG,
native SOMA->HumanML encoding and HumanML->SOMA reconstruction/resampling. No
shipped Python, hosted inference, procedural generation or pose-blending substitute.
All five real networks (CLIP, masked/residual transformers and RVQ encoder/decoder)
run natively. SMPL/SMPL-X, dataset motions, body meshes and evaluation models are
omitted. Existing native pose/root postprocessing enforces edit endpoints; source
poses outside the range are preserved, allowing normal quaternion roundoff/sign
continuity in exported GLB. Source transform roundtrips and real exported edits
are independently checked. Contact/velocity quality remains limited.

Windows ClangCL Release: eight CTests pass after correcting one CMake list fixture;
all 13 real native ORT component cases pass PyTorch reference bounds. RVQ indices
are exact. Full native generation token sequences match an independent reference
for walk/idle and contraction/punctuation prompts; official tokenizer IDs match.
Walk and idle edits preserve outside/endpoints within 1.20e-7 exported components
after quaternion sign matching. Two-job batches load sessions once. A two-job CPU
run takes 1.34 s including hashes/load, sampled peak working set 552.6 MB; generation
itself about .13 s/2-second clip. These are contended local observations. Cached
self-generated Kimodo comparison shows worse foot-slide/ground proxies for these
MoMask samples; no superiority claim. Bounded hidden Tide walk/wave captures from
engine checkout 28b61e4db at D:/Projects/PurrEngine pass and were inspected.

The owner's separate LOCALAPPDATA/TideMotion-MoMask-Experimental installation runs
with only Windows system directories on PATH and has its private verified bundle.
The existing TideMotion executable/models/PATH are preserved. No GPU work ran;
future GPU inference must hold TideMotionGpuValidation. No model weight, ONNX
graph, statistical array, BPE data, generated clip or screenshot is committed.

MoMask integration completion: rebased onto origin/dev 706665e, preserving native
ARDY, Kimodo measured selection and action policy, cleanup, and the MotionBricks
joint-frame/control update. Shared CMake, CLI includes and notes were combined
additively. The integrated Windows CPU Release build passes all nine CTests.
Matched upstream float32 cosine operation order at mask-count half boundaries;
a real 3-token/4-step native run and independent full-sampler reference agree,
and the default walk token sequence is unchanged. Windows case-insensitive path
collisions, blank prompts and preexisting partial files are rejected in preflight.
Weights remain private and the experiment remains off in ordinary release builds.

Final MoMask rebase also preserves origin/dev 68dd1c5's native upgrade command and
batched MotionBricks connects. Combined the additive upgrade/MoMask includes;
rebuilt the whole Windows Release tree and all ten integrated CTests pass. Native
model graphs, input pins, inference/encoding mathematics and asset validations
above are unchanged by this final integration. Refreshed only the separate local
experimental launcher/notices; the existing TideMotion installation stays intact.

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
Native output is the compact 30-joint rig; the 77-joint presentation skeleton is an
opt-in export since the Kimodo capability work (see that section).

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

Windows CI clang correction: the windows-2022 runner's ClangCL is newer than the
local 18.1.8 and enables -Wcast-function-type-mismatch, which rejected the direct
FARPROC-to-OrtGetApiBase cast in ardy_gpu.cpp under -Werror. The pointer bits are
now copied with memcpy (size asserted), which no cast-function-type variant flags.
Local ClangCL 18 full build and all twelve CTests pass. Only CI covers the newer clang.
The next run exposed that every Windows job since the upgrade test was added had
been cancelled before installation tests: upgrade.ps1 passed but ended on an
expected refusal, so its native nonzero exit leaked through $LASTEXITCODE into the
Actions pwsh wrapper. The script now ends with exit 0; the CI step reproduces exit 1
without it and exit 0 with it locally.

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
placement is validated empirically, not by upstream. (Dense paths, effector IK through
the MotionCorrection port, SOMA-77 output and --pose-map were added later; see the
Kimodo capability gaps section.)

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
model::load in this revision, so the environment is the interface. (Superseded: the
runtime patch wires them and the CLI passes options; see the Kimodo capability gaps.)

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

## Agent usage: choose checkpoints by action (2026-10-28)

Owner adopted action-specific checkpoint selection as the usage policy. AGENTS.md
now directs agents to the measured selection guide, requires explicit --model
(and batch model fields) for Kimodo authoring, and retains the asset plus recipe.
PROMPTING.md adds a bounded recommendation table, native install/generate examples,
a mixed-model batch, matched comparison procedure and grouping/reproducibility
advice. It identifies the original 0.5.0 measurements as v1.1 so they are not
mistaken for universal results across checkpoints. README exposes the selector
in the generate option table and links the agent-oriented instructions.

The recommendations retain their measured scope: longer plain right kicks,
curved-path error and crouch drift favor trying v1.2; backward/turn contacts and
the tested sword-worded kick favor v1.1. Other cases need candidate comparison.
The guides distinguish geometry/sparse-review evidence from guaranteed quality,
state that timings were not isolated, and keep ARDY usage separate. No automatic
router, runtime default, model/dependency/license pin or inference change is made.
Validation is documentation-only: local guide/report links and the example JSON
parse, native batch preflight accepts the documented options before the
expected empty-cache refusal, runtime staging carries updated guides/report, and
git diff --check passes. No inference, GPU allocation or model download is needed.

## Measured footfalls and pose-ranked selection (2026-10-28)

Follow-up owner request: close the step-count and raw-pose-fit workflow gaps.
Located an existing private sword idle at
D:/Projects/tide-motion-scratch/prompts/idle-soma.glb. Thus the earlier claim that
its source was unavailable is superseded. This file remains outside Git and is
not packaged. Native CPU reproduction (v1.1, 4 threads, 2 s, 50 diffusion steps,
seeds 1/3, start/end idle with endAt [0,-1]) gives end-pose raw rotation mean
15.88/16.62 degrees and maximum 53.59/51.09, matching the reported scale.

Investigated stronger rotation observation masks against the unchanged pinned
native model/source. Observing all 30 global rotations lowered mean error to
11.13/11.45 degrees but worsened end-root error from about 1.7 cm to 24.3/27.2 cm.
Observing just upper-arm/forearm rotations gave mean 16.08/16.07 and maxima
57.28/56.07 with root errors 5.3/7.1 cm. These are regressions for this workflow;
both experimental masks and their flags were removed. Existing model conditioning,
weights, dependency revisions, native endpoint correction and licenses are kept.
Raw error recipes now name rotationMaxJoint; the arm experiment's maximum was
LeftHandMiddleEnd, making terminal-joint orientation error visible rather than
silently hiding it from the existing all-joint metrics.

New explicit native selection controls:
- --footsteps N (0..64) accepts only a candidate with N measured travelling
  footfalls, trying consecutive seeds up to --attempts N (1..64, default 8).
- --fit-poses evaluates the complete attempt budget and chooses the lowest
  worst mean joint rotation error across full-body pose constraints, measured
  before correction. Requires a start/end/full-body keyframe. Combined with
  --footsteps, only candidates passing the count are eligible. Equal scores
  keep the earlier seed. No arbitrary guidance increase or altered pose mask.
- Both select one plain output, so --variants remains a separate operation with
  its existing seed-based names. Exhaustion writes no asset/recipe for that
  action and reports attempted counts; prior completed batch actions survive.
  All arguments/seed bounds/output paths are checked before model loading.
  Batch fields are footsteps, fitPoses and attempts. One RunContext/model load
  is retained. Model sampling alone generates every candidate; selection does
  not alter root travel, poses, timing or contacts to force a measured match.
- The saved seed is the selected candidate's actual seed. Selection metadata
  stores requested count (when used), firstSeed, attemptLimit, every attempted
  seed/count/raw score, and the fit ranking rule/selected score when used.
  Remake it with that seed and identical generation controls without selection.
  Ranking is a defined angular metric, not a claim of perceptually best motion.

Footfall measurement is a native SOMA FK state machine, rule
soma-travelling-footfalls-v1. Per foot, the lower tenth-percentile ankle/toe
minimum estimates ground height. Establish support for at least two samples /
0.06 s within 2 cm of that floor and with ankle horizontal speed <=0.8 m/s.
A completed swing clears 4 cm for at least two samples / 0.10 s and returns to
settled support at least 8 cm from its prior support. Durations round upward to
whole samples. Exclude incomplete initial/final swings and same-place kicks.
Report foot, lift, first settled landing sample, distance and peak clearance.
Sheet/GIF labels, inspect and saved recipes now carry count/events as appropriate.
The counter only measures; it does not infer IK contacts or run cleanup.
It is conservative on flat ground: tiny shuffles, sliding landings, stairs and
uneven surfaces can be missed; travelling hops can qualify as footfalls. Preserve
this explicit definition rather than claiming semantic recognition of steps.
Explicitly mapped custom rigs first retarget to SOMA; no automatic bone mapping.

Validation so far: Windows ClangCL 18 Release Vulkan build, all six CTests pass.
Authored independent support/swing/landing clips verify two alternating events,
settled landing frames, world translation/yaw invariance, same-place kick and
sliding exclusions, and unfinished initial/final swing exclusions. CLI/batch
checks cover numeric/type bounds, seed-budget overflow, incompatible variants,
missing full-body poses, orphan attempts, accepted controls and complete late
batch preflight. tests/selection-inference.ps1 provides opt-in real native
ranking/count/exhaustion, model reuse, exported endpoint and bounded sheet checks.
It coordinates CPU as well as GPU runs through TideMotionGpuValidation.

Successful CPU native count selection on the sword idle: seed 0 measured three
footfalls and was rejected; seed 1 measured two and was saved with actual seed
and attempted-count metadata. An impossible 64-footfall request evaluated exactly
two seeds (both measured two), returned nonzero and wrote neither GLB nor recipe.
The four prototype-batch actions (six candidates) reused one model load. Those CPU runs were
outside the GPU mutex while ARDY's CPU benchmark held it; they used explicit
--device cpu --threads 4 (no GPU inference), and the other task was informed so
its timing is reported as contended. Subsequent native measurement also acquires
the coordination mutex for CPU work. No throughput claims derive from these runs.

Limits: selection improves the recorded metric within a finite candidate set;
it does not retrain the model or promise a lower raw error than every possible
seed. Existing exact full-body endpoint correction remains in effect. Maximum
rotation/position errors stay reported alongside the ranking score so a lower
mean cannot hide a worse joint. Broad action/rig coverage and non-Windows builds
are not claimed here. No evaluation motion data or research-only weights enter
Git; no new model/runtime dependency was added.

Final native pose ranking ran on the CPU after acquiring TideMotionGpuValidation:
seed 2 worst pose mean rotation 17.3862 degrees; seed 3 16.8359 degrees. The
2-attempt --fit-poses action selected seed 3 and recorded both unrounded scores.
The native exported-GLB verifier passes for ranked and counted outputs, checking
all 30 local translations/rotations at both source/placed endpoints. Its optional
"keep" heading argument checks a two-coordinate endAt without imposing a new yaw.
All six CTests pass on the final pre-integration source. Bounded three-frame
TIDE_HIDDEN=1 sheet capture with the same engine-built Tide/platform succeeds at
960x998; labelled footfall counts and metrics were visually inspected. The
successful ranking/capture runs held the shared mutex. No timing claim is made.

Integrated onto origin/dev 93a92e3 (ARDY native backend/history, bounded cleanup,
HumanML and MoMask research retained). Combined shared CLI field/option sets so
--backend/--history/--history-frames and selection remain additive. The ARDY
candidate name uses the selected-output slot rather than attempt index, avoiding
an out-of-range output lookup on a second selection attempt. The integrated
Windows Vulkan build passes all eight CTests, including a new independently
predictable terminal-joint error diagnostic (90 degrees at LeftHandMiddleEnd,
zero position error, mean 3 degrees over 30 joints). Exported ranked/counted
endpoint checks pass again. Under TideMotionGpuValidation, the integrated ARDY
CPU executable with imported four-frame history, full-body start pose, --fit-poses
and an impossible --footsteps 64 budget of two generates both six-frame candidates,
reports zero footfalls and distinct raw pose scores, then correctly fails without
writing output. This exercises backend/history preservation and the second-attempt
output-slot fix with real inference; no ARDY quality or speedup claim is added.

Final integration also retains origin/dev 43f146c's paired SOMA-version comparison
suite and report. No runtime/source conflicts; append-only implementation notes
were combined. Reconfigured/rebuilt and all eight CTests pass on that final tree.

Retained 28ab94d's checkpoint-by-action guidance after the final docs-only rebase.
The selection example and new opt-in inference runner explicitly choose v1.1,
matching the actual measured backward-stumble runs and current AGENTS.md. The
C++ tree is identical to the final passing eight-test build.
## MotionBricks controls and SOMA joint-frame fix (2026-10-28)

Owner request: use the full capability of the NVIDIA models. Upstream read at the
unchanged pin NVlabs/motionbricks 54fc563aeaaf819a753d72cc9fdc6713989cdd43
(docs/onnx_inference.md, inbetweener.md, runtime_api.md, soma_model.md,
model/kernels.py, model/generation.py, model/soma/kernels.py, engine.py,
animation/foot_contacts.py) in ignored .research/upstream. No model, dependency
or ONNX Runtime revision changed. Kimodo gaps are handled in a separate thread.

**Joint-frame bug (fixed).** MotionBricks SOMA model space uses the BONES-SEED BVH
joint frames: bones lie along local +X (Spine1 offset 5.0 cm along X), so zero
rotation is not a standing pose. Canonical SOMA-30 (Kimodo) uses world-aligned
standard T-pose frames. Tide sent and decoded global 6D rotations without
converting, so the model received rotations off by roughly 90-120 degrees per
joint. Evidence from a dumped raw tensor (local debug build, not committed): raw
frames meant to reproduce the context differed by up to 123 degrees on every body
joint, and hip joints sat above the hips. Endpoint correction hid this at the
seams by blending up to 150 degrees (LeftLeg) through the bridge. Bridges then
showed tucked legs, with both ankles 0.3-0.55 m off the ground mid-transition.
Fix: Kimodo's own from_standard_tpose, G_bvh = G_std * offset, using the per-joint
offsets in nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954
kimodo/assets/skeletons/somaskel77/standard_t_pose_global_offsets_rots.p
(Apache-2.0, notice already in licenses/NVIDIA-KIMODO-SOURCE.txt). The raw float32
storage was read without unpickling; the SOMA-30 entries are compiled into
bricks.cpp as quaternions. Verification: Kimodo's somaskel77 bvh_joints reproduce
the MotionBricks SOMA BVH rest skeleton (Neutral_walk_forward_002__A057.bvh header
from git LFS, read only, ignored scratch, never copied) for all 77 joints with
0.0 cm difference. Encode applies the offset to every global rotation; decode
removes it. Positions, heading, alignment and endpoint correction are unchanged.
After the fix, with the same inputs, raw context frames match within millimetres
at the hip joints. Remaining raw differences above 13 degrees are only on hands
and hand end joints (up to 64 degrees). Those leaves carry Kimodo rotations that
move no other joint. Earlier GLBs from connect/pose-sequence should be regenerated.

**Controls added** (upstream names in brackets):
- Token ranges [allowed token counts]: `--tokens N|MIN..MAX`, batch/pose-sequence
  `tokens` integer or [min,max]; default stays 8..14. Recipes record allowedTokens
  and the model's chosenTokens. Pose-sequence timing is computed after generation
  (fixed counts still fill it at load, as before).
- Target pose count [Poses to Use / num_poses_to_use]: `--target-poses 1..4`,
  batch targetPoses, pose-sequence target `poses`. Masks follow
  write_motionbricks_conditions_kernel; unused slots repeat the last given pose
  (pad_token_poses_kernel). `poses` and legacy `hold` are exclusive.
- Early transitions [Transition Early By]: pose-sequence non-final target `early`
  frames. The bridge is cut there and its last four played frames become the next
  context, as in the Inbetweener. At least four frames play, so early <=
  4*min tokens-10. Guide arrivals are not exact and are excluded from protected
  frames (main.cpp and tests/verify_connect.py).
- Predicted foot contacts: the last four output features per frame (left heel/toe,
  right heel/toe; clamped 0..1 as upstream). Recipes always record
  predictedFootContacts. `--contacts-out PLAN.json` / contactsOut write planted-heel
  runs (>= --contact-threshold, default 0.5 as upstream's renderer, >= 3 frames)
  as an ordinary cleanup plan. Heel is used because cleanup anchors the ankle.
  Nothing is cleaned automatically.
- seamCorrection in connect recipes: raw hips miss per seam and largest joint
  correction in degrees. This diagnostic exposed the frame bug above.

Model-free tests: default/range/fixed token masks and parsing refusals, masks for
1..4 target poses, five poses refused, contacts trimmed/ordered/clamped/nonfinite
refused, planted-run extraction and offset, encode->decode round trip with zero
seam correction, Spine1 model x axis up at rest, pose-sequence parsing of
ranges/poses/early and refusals (final early, too early, poses 0, hold+poses,
inverted or out-of-range tokens), timing waits for the model with ranges. CLI
tests: option refusals, contacts-out path rules (.json only, not the recipe),
batch refusals and plan-path collisions, accepted options reaching model
verification. Windows x64 ClangCL 18 Release: all CTests pass (eight after
rebasing onto the native ARDY and footfall-selection work; the runner below was
repeated with that integrated build and passes).

Real native ONNX CPU inference (8 threads, TideMotionGpuValidation held), using
user-generated Kimodo idle/walk clips (walk moves about 2 m forward). The target
was the walk clip offset 3 m, so its start lies about 1 m past the source's end.
Measurements are hips ground speed over the last three intervals and seam numbers
from the recipe:
- Before the fix: tucked legs (lowest foot 0.2-0.38 m mid-bridge), seam joint
  corrections 150-171 degrees, and no heel run reached 0.5 for 3 frames.
- After the fix: lowest foot 0.00-0.03 m above the floor for the whole bridge,
  raw hips miss at most 2.6 cm, worst seam joint about 80 degrees, only on
  RightHandThumbEnd.
- Poses to Use: into the walking target, 1 target pose arrives at 0.12 m/s,
  against 0.96 m/s (2 poses) and 0.95 m/s (4 poses), so 1 stops as documented
  (0.004 m/s into idle). Before the fix 1 pose also stopped (0.14 m/s against
  1.16-1.21).
- Predicted contacts against measured foot speed: Pearson r -0.65 to -0.82 per
  heel/toe. Flagged feet move 0.09-0.26 m/s, unflagged 0.7-2.2 m/s. Each bridge
  yields 3 planted-heel intervals.
- tests/pose-inference.ps1 (extended: guided example
  examples/pose-sequence-guides.json with ranges/poses/early and contactsOut,
  walk->walk ranges, poses 1 vs 4 with asserted arrival speeds, raw hips seam miss
  < 10 cm, cleanup with both predicted plans) passes with one ONNX session for
  six actions. The guided sequence verifies 2 exact endpoints, 1 guide and its
  timing. Cleanup of a predicted walk plan applied one interval and skipped two
  (drift-increased, joint-step-increased). Those spans already moved about
  0.004 m/frame.
- tests/cleanup-inference.ps1 (connect pins plus cleanup, independent Python
  verifier) passes.
- Bounded TIDE_HIDDEN=1 captures (60 frames) of the guided sequence and the joined
  walk show upright mid-stride poses with planted feet. They used Tide built from
  D:/Projects/PurrEngine 28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c, staged privately
  (copied into this task's ignored .research). That staging predates tide run
  --record, so preview --gif was refused and not used. No engine change.

Gaps: batched N>1 requests (upstream supports N; Tide runs one request at a time,
and that is only a speed matter on CPU); upstream IK Pins (TargetPoseIKCorrector;
Tide's explicit cleanup is the alternative); Smart Primitive and scene editors
(object interaction placement); runtime blend-space steering; FP16 model variant
(no CPU benefit). In these runs the model picked the longest allowed count (15-16
tokens) for a 1 m gap. Tide does not override the model's choice. Hand and leaf
rotations still disagree at seams as noted. Predicted contacts are model output,
not ground truth, and are not tested on non-walk motions. Linux/macOS untested.

## Native executable upgrade path (2026-10-28)

Owner request: a quick `tide-motion upgrade` path like Tide's. Inspected Tide's
compiler/cli/upgrade.c and install.c in D:/Projects/PurrEngine (read-only) for
latest/exact version selection and replacement of mapped Windows programs.
Implemented independent native C++ orchestration in src/upgrade.cpp/.hpp; Tide
Motion still requires no engine installation, Python, GitHub CLI or account.
No inference, model or dependency revisions changed, and no model files/data
were added. The updater uses the existing SHA-256 and argv process functions.

Commands: upgrade defaults to the latest published stable GitHub release;
--check reports the selected version/platform without changing an installation;
--version X.Y.Z (optional v prefix) explicitly selects/reinstalls that published
stable version, including older versions; --force reinstalls the default/local
selection even when equal/older; --archive FILE uses a local package plus adjacent
.sha256 without network; --root DIR explicitly selects an installation root,
including an empty one. Local archives and exact-version options are exclusive.
Default root comes from the real running executable's bin directory and requires
an installed license layout, never the current working directory. Source builds
need --root to install. No-op latest/equal versions do not replace files.
No model-cache environment, model download, PATH change or elevation is involved.
No nightly channel, background check or dev-artifact auto-install is claimed.
Existing executables lacking upgrade need one initial manual installation of a
build/release with this command. No version tag/release was published here.

Network contract follows GitHub's public REST latest/tag API:
https://docs.github.com/en/rest/releases/releases#get-the-latest-release .
Numeric stable tags only, published/non-draft/non-prerelease. Select the exact
archive and sidecar names for windows-x64, linux-x64/arm64 or macos-arm64; reject
missing/duplicate assets, foreign download URLs and invalid sizes. Initial asset
URLs must match this repository's exact release/tag/name; curl -q uses HTTPS-only
initial/redirect protocols, bounded timeouts/retries and no shell interpolation.
Metadata <=1 MiB, sidecar <=4 KiB, archive <=512 MiB. Require matching published
size and SHA-256 before extraction. Checksums are corruption/integrity checks
from the same trusted HTTPS release, not independent publisher signatures.

Extraction is into a private destination-volume staging directory. Windows uses
a fixed locally written PowerShell Zip extraction script (never a downloaded
script): refuse traversal, absolute/drive/ADS paths, symlinks, duplicate/case
aliases, reserved device/trailing-dot/space names, >10,000 entries and >512 MiB
uncompressed bytes. Unix uses system tar with bounded member/type listings;
refuse traversal/backslashes/absolute paths, hard/device links and all symlinks
except bounded sibling libonnxruntime.* names used by the distribution. Unix does
not claim an uncompressed-size cap before extraction. Validate required package
entries and licenses, execute staged --version and require exact release identity
before changing existing files. PROMPTING.md is optional for pre-0.5.1 releases;
if absent in an explicitly selected older release, its prior distributed copy
is removed transactionally rather than falsely retained as current documentation.

A per-root OS lock (Windows exclusive handle / Unix flock) serializes installs.
Rename only distribution-owned bin, licenses, examples, notes, README, PROMPTING,
LICENSE and installer scripts. Models, settings and other root files are never
moved. Keep all replaced entries in a separate marked backup; reverse every swap
on reported failure. If rollback itself fails, retain originals and name their
backup location. Windows can rename mapped runtime files; any locked old files
are retained after success and collected on a later upgrade only when marked
committed. Pending/interrupted recovery backups are never automatically deleted.
Cleanup is bounded to verified direct child scratch/backup directories. This is
rollback for reported errors, not crash/power-loss atomicity across all files;
--force is available to repair/reinstall. README records these limits.

Local Windows x64 ClangCL 18.1.8 Release: all eight CTests pass. New independent
release contracts cover numeric ordering, malformed versions, four platform names,
drafts/prereleases, foreign URLs, oversized/invalid assets, missing checksums and
duplicate assets. CLI contracts cover help/options and invalid version/flag
preflight without a model cache. tests/upgrade.ps1 takes a runtime archive and
creates a GUID installation with spaces/apostrophes: offline check leaves root
absent; offline install and self-upgrade from the running installed EXE succeed;
models/settings remain; locked LICENSE forces rollback after preceding metadata
swaps, preserving README/note sentinels and executable; exclusive install lock
refuses a second update; corrupted hash, traversal and executable/version mismatch
are refused. Committed-backup cleanup and pending-backup retention pass.

Real online path: `upgrade --check` used GitHub's live latest API, which selected
v0.2.0 at validation time. `upgrade --version 0.2.0 --root .research/online-upgrade`
downloaded its actual official 7.50 MiB archive/sidecar, verified size/hash, probed
its executable and installed successfully in isolation. No owner installation or
model cache was changed. This exposed/closed the legitimate legacy PROMPTING
absence. The live updater deliberately refuses to downgrade a newer executable
via the default latest route without --force; exact-version selection is explicit.

Ubuntu 24.04 x64 / GCC 13.3 under WSL: fresh native CPU build, all eight CTests,
runtime archive and tests/upgrade.sh pass, including offline/live self-replacement,
model/settings preservation, rollback on a late symlink destination and bad
checksum/escaping-symlink refusals. A GCC misleading-indentation error encountered
in the existing native v1.2 converter was fixed by putting output.close() on its
own line; no converter arithmetic changed. tests/upgrade.sh has LF line endings.
macOS/ARM64 execution remains unvalidated locally; platform selection is covered
by contracts, with native execution left to the existing release matrix.

Release workflow correction: its explicit build target list omitted newer CTest
executables (cleanup/HumanML and now upgrade), so packaging could not reach a
successful contract stage. Build all default targets before ctest. Existing
packaging/install jobs now also run native self-upgrade integration scripts on
their actual archives. Action/dependency pins and publication gates are unchanged;
no additional CI dispatch, PR, model inference, GPU work or engine preview was
needed. Tests/packages/live downloads remain ignored under .research or the
WSL-local build. No package executables or model data enter Git.

Upgrade integration completion: rebased onto origin/dev 706665e, retaining the
native ARDY backend/contract tests, measured motion selection, updated MotionBricks
pose/guide/contact support, notices and all concurrent documentation. Resolved
additive CMake/main include/CLI-test/notes conflicts. Windows and WSL Linux x64
Release builds and all nine integrated CTests pass. Both integrated archives pass
self-upgrade tests again; Windows's existing installer test also passes. Workflow
YAML parses. No action pins, publication gates or extra workflow dispatch changed.
No model/backend inference was repeated for this executable-maintenance feature.

Final hardening: local archives are copied into owned scratch before hashing and
extraction, so the extracted bytes are the verified copy. POSIX scratch directories
use owner-only permissions. All subprocess paths use UTF-8 at the existing process
API boundary; Windows tests now put both the archive and installation under paths
with spaces, apostrophes and an accented Unicode character. Current upgrade unit
and both platform archive/self-replacement tests pass after these changes. This
also verifies the equal-version no-op retains a modified documentation sentinel
until --force is requested. ARM64/macOS execution remains a release-matrix check,
not a local claim. No release tag/publication or real owner-install modification
was performed; bootstrapping older installed binaries remains a first manual
installation of a future published package containing the command.
## Batched MotionBricks connects (2026-10-28)

Owner request: add batching. Upstream docs/onnx_inference.md (same pin
54fc563aeaaf819a753d72cc9fdc6713989cdd43): every input takes N independent
requests, concatenated along axis 0. Each request is trimmed, corrected and
placed separately. BricksModel::generate(span<Request>) encodes each request,
concatenates the seven inputs, runs one session call, then decodes and
endpoint-corrects each [64,366] slice with its own alignment and guides. The
single-request API now goes through it with N=1. Limit 32 per run
(bricks_batch_limit). No model, ORT or dependency revision changed.

Batch integration reuses the normal connect path unchanged. On reaching a connect,
batch() runs the following consecutive connects in a prepare mode. That does all
argument parsing, path, preflight, metadata and clip validation, then records
(from, offset target, options, output) without inference. The run stops at a
non-connect, a different --threads, the limit, an action whose from/to is an
output of an earlier action in the run, or an action whose preparation fails
(it then runs normally later and reports its own error). Runs of two or more
requests use one session call. The normal executions then take their bridges in
order and check the output path. Joins, pins, contacts and recipes are as before,
plus motionbricksBatch {requests, index}. A failed batched run writes none of its
outputs; earlier outputs are kept. stderr reports the actions grouped into each
run and the model time per call ("MotionBricks inference: S s").

Validation, Windows x64 ClangCL 18 Release: all CTests pass (nine after rebasing
onto the native upgrade work; the runner below was repeated there), including a
new CLI contract that two connects form one run (reported as "Batch actions 1-2"
when models are missing). Opt-in tests/bricks-batch-inference.ps1, holding
TideMotionGpuValidation, used user-generated Kimodo idle/walk clips. It ran 32
varied independent connects: offsets, poses 1..4, fixed and ranged tokens, and
joined or bridge-only output. It compares them with the same 32 run one per run in
one process (separated by mirrors). Results:
- one ONNX session and one 32-request call;
- identical chosen tokens;
- maximum local difference 1.86e-6 m and 6.9e-4 rad (CPU kernel blocking).
A dependent chain (third action reads the first's output) split into runs 1-2 and
3-4 as designed. Model time for 32 connects: 0.570 s batched against 0.994 s one
per run at 8 threads; 1.730 s against 2.096 s at 1 thread. Wall time
3.78 s against 4.33 s, dominated by session load and GLB I/O. These are single
runs on this machine, not a benchmark.

Gaps: no batching across pose-sequence segments (each uses the previous
segment's generated context) or across separate pose-sequence actions. Connect
actions that consume an earlier connect in the same run are not batched. No GPU
provider (MotionBricks stays ORT CPU). The batch limit is fixed, not configurable.


## Owner-requested local Windows update (2026-10-28)

Updated the owner's existing %LOCALAPPDATA%/TideMotion installation from a newly
built Windows x64 Vulkan Release package of b59b270. Reconfigured this checkout's
build with MOTION_VULKAN=ON and the pinned installed Vulkan SDK 1.4.363.0, rebuilt,
and passed all nine CTests with isolated TEMP/TMP and TIDE_HIDDEN=1. PE dependency
inspection confirms the Vulkan delay-load import, preserving the previous GPU
build capability rather than replacing it with the earlier CPU test package.
No GPU inference, model download or benchmark was required for this maintenance.

Packaged the runtime, generated its SHA-256 sidecar and used the new native
`upgrade --archive ... --root <installed root> --force` path. Installed --version
and help upgrade succeed. The installed executable matches the validated build's
SHA-256 2a3b1a41da258e9974206ff5c8ebc3b109458a2bc358c0a70377afb30bcb9754.
All four existing model files retain their paths, sizes and last-write timestamps;
no multi-GB model checksum/inference run was repeated. User PATH is unchanged and
already contains the installed bin directory. The package includes current
licenses, authoring examples and documentation. The displayed version remains
0.5.1; b59b270 identifies the local development code installed. No release was
published, engine files modified, or user settings/models changed. Build/package/
verification records remain ignored under .research/local-update-*.

Two concurrent commits landed during the first build. The final installation was
therefore refreshed once more from the integrated adf5567 code (including native
MotionBricks request batching), with Vulkan still ON and the separate MoMask
experimental build option OFF. All nine Vulkan CTests pass again. Repackaged,
checksum-verified and installed through the same native upgrade path. Final
installed executable SHA-256:
750c773b515212208d2d95c3707d40f7c28d4f33bd06dc2df42f69d14e5c7c1a.
Installed --version/help upgrade and binary identity checks pass; all four model
file metadata records and user PATH remain unchanged. The owner's separate
TideMotion-MoMask-Experimental installation is not touched. This documentation
record does not change executable code or require another inference run.
## Kimodo capability gaps: SOMA-77, dense paths, MotionCorrection, runtime options, mapped poses (2026-10-28)

Owner goal: use the rest of what NVIDIA Kimodo offers, natively. Upstream read-only at
nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 (ignored .research clone);
kimodo.cpp and GGML pins unchanged. No Python, service, procedural generator or new
runtime dependency. No model, evaluation motion data or research-only checkpoint used
or committed (upstream's demo motion.npz files were not opened).

### 1. SOMA-77 presentation skeleton (closed, opt-in)

kimodo.cpp returns only the 30 predicted joints (its README lists 77-joint expansion as
not implemented). Upstream's expansion (SOMASkeleton30.to_SOMASkeleton77 /
output_to_SOMASkeleton77) is a slice insert: the 30 local rotations are kept and every
other joint takes a constant relaxed-hands rest rotation; from_SOMASkeleton77 slices the
30 back. All 30 SOMA-30 joints have bit-identical neutral positions in the 77 joints.p.
src/soma77.hpp is generated from the Apache-2.0 repository assets
(somaskel77/joints.p as parent-local offsets, relaxed_hands_rest_pose.npy as unit XYZW
quaternions; names/parents from definitions.py). They are skeleton definitions in the
Apache-2.0 code repository, not model checkpoints or motion data; the existing
NVIDIA-KIMODO-SOURCE notice covers them. Regenerate with the developer tool
tools/generate_soma77.py UPSTREAM_CHECKOUT (stdlib zip/pickle with a tensor-only
unpickler and NumPy; not shipped or built), which also reports left/right asymmetry: offsets 2.4 mm, relaxed
rotations up to 0.24 in matrix entries (the relaxed pose is not mirror symmetric).

Contract: `generate --rig soma30|soma77` (batch `rig`), default soma30. Expansion is done
at export after correction/in-place, so the motion equals the SOMA-30 result. Every
canonical reader (connect, constraints, pose-sequence via load_glb, retarget source,
cleanup/mirror) recognises a GLB with LeftHandIndex1 and reads its exact SOMA-30 subset
(names/hierarchy/unit scale/proportions within 1 mm checked; animated bone lengths
refused). mirror keeps 77 joints (body as mirror_soma, other joints take their
counterpart's M L M); cleanup cleans the subset and keeps the clip's own non-subset
rotations. inspect reports `soma77`. preview shows the subset on the mannequin or a
character (`--map` uses SOMA-30 names) and all 77 with `--skeleton`. connect, retarget,
pose-sequence and constraint use write SOMA-30. Fingers are not animated by the model;
ThumbEnd/MiddleEnd are among the 30 predicted joints and do vary.

### 2. Dense root paths with upstream smoothing (closed)

Upstream's demo dense path (viz/constraint_ui.py RootKeyframe2DSet, demo/generation.py):
lock keyframes at frame 0 (and the end), linearly interpolate the 2D waypoints to every
frame of their span, then with 3+ keyframes smooth with motion_rep/smooth_root.py
get_smooth_root_pos (multigrid ADMM, 6 cm margins, pos_weight 0, over-relaxation 1.8, 500
iterations per level); constrain every frame (root2d), and give full-body/effector
constraints the dense sample as smooth_root_2d. src/constraints.cpp now has
smooth_root_path (a banded LDL^T solve replaces scipy splu), smooth_root_of (the
representation's smooth root of an output) and densify_path. `--dense-path` (batch
`densePath`) applies it to the --path waypoints, Kimodo only. Frame 0 is added at the
start pose's hips or the frame-0 path point, else the origin; the dense span ends at
the last waypoint (later frames stay free: Tide has no previous motion to lock the end
to). Waypoint yaws are kept (upstream's dense mode drops headings; an extension).
Interpolated samples neither place nor turn poses; a pose on the path is encoded with the
sample as its smooth root. Recipes keep the waypoint errors and add densePath
{first/last frame, frames, smoothed, smooth-root and hips error mean/max}.

Parity: tests/smooth_reference.py transcribes upstream smooth_signal with NumPy (dense
solve) and drives the native code through `capability-tests --smooth`: line, corner,
137-frame random walk and 6-frame cases agree to 1.1e-12 m. Upstream behaviour kept on
purpose: its final step-1 averaging shifts the result by half a frame, and 500
iterations from a mean initialisation pull a constant-speed line toward its middle by up
to 6 cm (8 cm with the shift). Paths shorter than five samples are returned unchanged
(upstream's coarsest system would be singular).

### 3. Native MotionCorrection port (closed as opt-in; measured against Tide's correction)

src/correction.cpp ports TrajectoryCorrector, InverseKinematics and Utility CorrectMotion
plus the BindingsPython/postprocess.py argument setup (working rig 2 cm above ground,
thresholded contacts > 0.5, LeftHand/RightHand/LeftFoot/RightFoot pins with upstream
hint offsets, ankle two-bone and toe one-bone contacts, root margin 0.04). Scalar C++ on
Tide's types; SIMD, Eigen and pybind11 are not used. Upstream's products are in
DirectXMath order (a*b applies a first); each was rewritten in Hamilton order. Kept
upstream quirks: velocity weights never reach V (multVelWeights discards its result), the
left toe reuses the right toe's minimum height, the opposite-vector fallback quaternion
is upstream's, one-bone toe intervals touching a constrained frame are dropped (so a
start pose removes toe locking in an all-contact standing clip). Not ported: the G1
double-ankle phase (SOMA has single ankles) and per-segment post-processing inside
multi-prompt sampling (Tide corrects the joined output once, after sampling, so later
segments are not conditioned on corrected motion).

Contacts come from the model: patches/kimodo-runtime.patch adds motion_data.foot_contacts
(the last four decoded body features, unnormalised, both samplers). `--correction
tide|kimodo|none` (batch `correction`); `--no-correction` stays as `none`, default tide is
byte-identical to before. kimodo needs --backend kimodo and refuses --correction-frames.
Recipes add `correction`, `kimodoCorrection` (source, margin, threshold, contact frame
counts, effector/ankle/toe IK frames, milliseconds) and, for every Kimodo generation,
`footContacts` intervals [start, end) per channel.

Upstream parity: tests/kimodo-correction-parity.cpp is a developer-only driver that runs
NVIDIA's own MotionCorrection C++ (MSVC, /arch:AVX, Eigen 3.4.0 3147391d at the tag
upstream fetches, all in ignored .research/parity) on inputs traced by the native port
(correct_kimodo's optional trace stream). `capability-tests --replay RAW.glb RECIPE.json
TRACE OUT.glb` rebuilds the inputs from a `--correction none` clip and the kimodo
output's recipe (same seed, so identical raw motion and contacts). Synthetic case with
every stage active (correction moved the root 10 cm, rotations up to 81 degrees): root
identical, rotations max 0.040 / mean 0.0001 degrees. Real clips (12: walks, kicks
between idle poses, a path walk and a right-hand effector reach, seeds 1-3): root
difference 0 m, rotation max 0.040..0.079 degrees, mean <= 0.0003 degrees (float
rounding); the offline replay reproduces the tool's output to 0.055 degrees.

Measured comparison (RTX 4080 SUPER, Vulkan, v1.1, 50 steps, guidance 2, seeds 1-3 each;
the same raw motion per seed; positions from independent GLB FK in
tests/capability_metrics.py; skate-from-contacts follows upstream FootSkateFromContacts:
ankle/toe speed while the model predicts that contact; skate-from-height follows
FootSkateFromHeight: toe speed while the toe is below 5 cm):

| Case | Correction | Pose error after | Effector / path after | Skate contacts m/s mean (peak) | Skate height m/s | Max joint step deg |
| --- | --- | --- | --- | ---: | ---: | ---: |
| walk 3 s | none | - | - | 0.048 (0.434) | 0.477 | 17.5 |
| walk 3 s | kimodo | - | - | 0.029 (0.298) | 0.602 | 17.5 |
| kick 2.5 s idle->idle | none | 1.2/2.6 cm mean start/end | - | 0.055 (0.273) | 0.108 | 33.0 |
| | tide | exact | - | 0.064 (0.273) | 0.119 | 33.0 |
| | kimodo | exact | - | 0.030 (0.345) | 0.165 | 32.9 |
| path walk 3 s | none / tide | - | hips 6.8 cm mean, 9.3 max | 0.043 (0.196) | 0.339 | 13.3 |
| | kimodo | - | hips 3.9 mean, 4.0 max | 0.023 (0.449) | 0.475 | 13.3 |
| reach 3 s, effector @60 | none | start 1.2 cm | hand 1.5 cm | 0.016 (0.095) | 0.017 | 54.3 |
| | tide | exact | hand 1.5 cm | 0.018 (0.091) | 0.019 | 54.3 |
| | kimodo | exact | hand 0.0 cm | 0.032 (0.373) | 0.035 | 54.6 |

Kimodo correction took 0.9-2.0 ms per clip. Reading: it makes effectors exact, holds
path frames within the 4 cm margin and roughly halves sliding where the model predicts
contact, but it raises the height-based skate number (more toe samples are forced to the
floor and slide sideways, as upstream allows), can pop a foot at contact boundaries
(peaks to 0.45 m/s) and doubled a standing clip's sliding (its root follows the effector
margin while ankles are pinned with reach falloff). Tide's fade correction does not touch
contacts and slightly raises kick sliding. Neither is better everywhere, so tide stays
the default and kimodo is opt-in. Foot contacts in Tide's separate `cleanup` remain
explicit plans; `footContacts` intervals can inform them.

### 4. kimodo_runtime_options (closed)

The pinned C API validated kimodo_runtime_options' size and then ignored it; the C++
model::load had no options. patches/kimodo-runtime.patch (second patch, applied after
kimodo-conditioning.patch; cmake/apply-patch.cmake now restores the pinned tree and
applies the ordered list, so repeated or changed populates are idempotent) adds
kimodo::runtime_options {device automatic|cpu|vulkan, threads} and
model::load(motion, text, options), threaded to the deferred motion-weight load and the
text encoder. automatic keeps the old environment behaviour; cpu never touches Vulkan;
vulkan fails rather than falling back; threads 0 keeps KIMODO_THREADS or the hardware
count. kimodo_model_load maps the C struct and now refuses a non-empty backend_dir
(backends are static) and unknown devices. tide-motion passes --device/--threads as
options and no longer sets KIMODO_BACKEND or KIMODO_THREADS; an operator's
KIMODO_BACKEND=cpu (auto) and KIMODO_THREADS (threads 0) are still honoured.
KIMODO_TEXT_LAYER_CHUNK and the other tuning variables have no runtime-option field and
stay environment variables. Evidence: the default GPU output of `generate "A person
walks forward" --duration 3 --seed 1 --no-correction` is byte-identical between the
installed 0.5.1 binary (environment variables) and this build; `--device cpu` with 2 and
8 threads gave identical payloads in 28.4 s vs 19.5 s (1 s, 10 steps, whole command).
capability-contracts covers the C API refusals and option pass-through.

### 5. Constraint poses from mapped humanoids (closed, with retarget's limits)

`--pose-map BONES.json` (batch `poseMap`): a pose file that is not SOMA-30/77 (any mapped
GLB or FBX) is retargeted to canonical SOMA exactly as `retarget --to-soma` (rotation
and root retargeting with the mapping's calibration; non-root translations dropped and
reported). SOMA files are still used directly. Recipes add the mapping and the list of
mapped sources. The constraint is the retargeted SOMA pose, so Synty-style calibration
differences (the leaning spine noted in 0.4.0) carry into it. Real run: Synty
A_Idle_Standing_Femn.fbx frame 0 with examples/synty-polygon.json as the start of a 2 s
walk: model error mean 1.87 cm / max 5.97 cm, rotations mean 13.3 / max 48.2 degrees,
then exact after tide correction. pose-sequence (author.cpp) is unchanged and still needs
SOMA sources.

### Validation

Windows x64 ClangCL 18 Release with Vulkan (warnings as errors): all ten CTests pass,
including the new capability-contracts (SOMA-77 rest/roundtrip/reduction/mirror/template,
dense smoothing/densify/conditioning, correction exactness/margins/contact IK/effector
reach, C API options) and CLI/retarget contracts for every new option, batch field and
SOMA-77 command path. Ubuntu 24.04 GCC 13 under WSL, fresh fetch through both patches,
builds with -Werror and passes all ten CTests. Developer checks (not CTests):
tests/smooth_reference.py, tests/kimodo-correction-parity.cpp,
tests/capability_metrics.py.

Real native inference: tests/capability-inference.ps1 (opt-in; holds
TideMotionGpuValidation; user/self-generated idle; optional mapped character). Run 1: one
batch of 49 actions (Kimodo loaded once; 74 s) on the RTX 4080 SUPER: the correction
cases above, sparse vs dense paths, SOMA-77 walk/connect/mirror/kick-from-77/retarget to
the Synty character, the mapped start pose, CPU thread timings and the baseline. Run 2
(34 actions, 60 s) repeated the correction cases with `footContacts` for the parity and
contact metrics. SOMA-77 `walk77` equals the same-seed SOMA-30 walk: hips and every body
joint 0 m apart, local rotations within float rounding; connect from it wrote a 174-frame
SOMA-30 sequence, mirror and kick-from-77 wrote 77 joints, retarget wrote the 50-bone
Synty rig. Dense versus sparse over three S-curve waypoints (4 s, seeds 1-3, no
correction): hips at the waypoints 10.8 cm mean / 22.1 max sparse, 5.2 / 10.5 dense; dense
smooth root 2.4-3.1 cm mean / 4.8-6.2 max from every path frame (hips 4.9-5.6 / 9.5-10.9);
dense + kimodo holds hips within 4.0 cm of every path frame (mean 3.0-3.2). Dense paths
did not change sliding consistently (height skate 0.350/0.428/0.421 sparse vs
0.335/0.531/0.479 dense). Bounded TIDE_HIDDEN=1 960x540 captures (frame budget 45) of
walk77, kick77, walk77-to-idle, dense-1, reach-kimodo-1 and walk77 on the Synty character,
with Tide from D:/Projects/PurrEngine 28b61e4db (the clean checkout; its privately staged
build copied into ignored .research/engine), were inspected: plausible poses, feet at the
floor, no binding errors or inverted limbs. Sparse visual checks only. Synty assets stayed
in their local pack; nothing generated or third-party is in Git.

Integration: rebased onto origin/dev adf5567 (MoMask, batched connects, upgrade, footfall
selection, ARDY), keeping their options; all ten CTests pass again, and one mutex-held
integrated smoke (4 s walk from the idle, dense S-curve, --correction kimodo, --rig
soma77) produced 120 frames: smooth root 2.9 cm mean / 4.6 max from the path, 119 ankle
and 103 toe contact frames corrected in 2.9 ms.

### Remaining gaps

- Kimodo correction is per whole output, not upstream's per-segment correction inside
  multi-prompt sampling; it does not run for ARDY. Its contact-boundary pops and the
  standing-clip regression are upstream behaviour reproduced faithfully, not fixed.
- Dense paths end at the last waypoint (no end lock from a previous motion) and are
  Kimodo only; yaw is an extension upstream's dense mode does not use.
- SOMA-77 fingers are upstream's constant relaxed pose; Kimodo predicts no fingers. The
  mannequin has no fingers; previews of SOMA-77 show the 30-joint subset unless
  --skeleton is used. connect/retarget/pose-sequence output SOMA-30.
- --pose-map inherits retargeting's limits (calibration, dropped non-root translations,
  humanoids only) and is not available to pose-sequence; history (ARDY) still needs SOMA.
- Linux/macOS inference, CPU inference beyond the thread timing, and a v1.2 comparison of
  the correction were not run; macOS/ARM64 builds are left to CI.

### 2026-10-28 — ARDY native Windows GPU acceleration

Owner requested GPU support after the completed native CPU backend. Implemented
real DirectML inference for the unchanged pinned denoiser, FSQ encoder and decoder;
the native sampler/history/mask/origin loop and canonical SOMA GLB export are
shared with CPU. CPU text embedding remains native/local GGML. No weights,
evaluation data, research checkpoints, Python environment, hosted text service,
game simulation or procedural/blended replacement enters the delivered tool.

Contract: `generate --backend ardy --device gpu` requires the GPU path; `cpu`
forces CPU; `auto` tries GPU and logs CPU fallback if initialization or native GPU
execution fails. A run-time GPU failure restarts the entire generation from the
same seed on CPU, discarding partial GPU motion. Only GPU execution failures
trigger this retry, not prompt/asset/text validation failures. Cached sessions are
reloaded when requested device/threads change; old sessions are released before
allocating replacements. CLI and batch recipes record actual `device` and
`accelerator:directml|cpu`. Default Windows x64 builds stage/install three DLLs
under `bin/ardy-directml`; `MOTION_ARDY_DIRECTML=OFF` retains a CPU-only build.
Other platforms do not gain a GPU backend from this change.

Native dependency pins/terms are detailed with primary links in
`notes/ardy-native.md`: Microsoft.ML.OnnxRuntime.DirectML 1.24.4 (source
2d924974ef147392ced8409d36bd6d2e7fcc8a74, archive SHA-256
57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681),
Microsoft.AI.DirectML 1.15.4 (SHA-256
4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9).
ORT MIT/third-party notices and DirectML proprietary Windows redistribution terms
and notices ship intact. DirectML section 1(a) permits distribution inside these
Windows applications. Checkpoint/license pins remain unchanged from native CPU.
The Microsoft GPU binaries add 35,877,968 uncompressed bytes; no user SDK/CUDA/.NET
installation is needed. DirectML requires a suitable DX12 driver/Windows version;
upstream sustained-engineering status remains a maintenance risk.

CPU ORT stays at 1.31.1. The optional GPU TU explicitly loads a separate ORT 1.24.4
C API table from its absolute executable-relative directory; modules outlive API
objects. Numeric vectors alone cross that boundary, with no ORT tensor/allocator
or C++ API-global mixing. Sequential sessions/disabled memory patterns follow
DirectML requirements, and ORT telemetry is explicitly disabled. Actual adapter
name and sampled per-process local DXGI memory are logged. Optional developer
TIDE_ARDY_PROFILE collects provider placement: the recorded profiles contain 43
denoiser / 7 decoder / 3 encoder DML kernel events and zero CPU-provider kernel
events (copy/barrier events may have no provider). This is executed native GPU
evidence, not an ONNX-export/session-load claim.

Validation: Windows x64 ClangCL 18.1.8 /W4 /WX Release, RTX 4080 SUPER 16 GB,
driver 617.42, Ryzen 9 7900X, eight CPU threads. All eight model-free CTests pass
in GPU-enabled and separate CPU-only builds. Native reference tests passed all
three networks at history 0/4/240, four-chunk 183-frame path-constrained host
generation, imported 64-frame history, and native preprocessing/FSQ. Max tensor
errors: denoiser 0.000138, encoder 4.77e-7, decoder 0.001622. Original tensor
assertion (0.002 + 0.002*abs(reference)) is retained. Decoded GPU host bounds are
5 mm / 0.025 radians versus CPU 3 mm / 0.01 radians to account for measured
discrete-FSQ sensitivity; actual fixed-input host 3.542 mm / 0.013655 radians,
imported fixed-input 0.258 mm / 0.003383 radians, native history preprocessing
max feature error 1.789e-5 and full imported path 2.173 mm / 0.032190 radians.
GPU/CPU bitwise equivalence is not promised.

Matched five-action batches with the same prompts/seeds/model/no-correction:
CPU/GPU seconds respectively: cold 183 frames at 10 steps 18.0096/14.6731;
warm same prompt/seed 5.23834/0.221005; imported-history prompt change plus path
123 new frames 4.50332/0.575913; sparse full-pose/hand/path constrained 183 frames
4.92461/0.220701; warm 1,800 frames at four steps 14.8352/0.699501. Generation
times include local text and exclude session creation. Whole-process wall times
49.4803/19.6703 seconds; peak 250-ms-sampled host working sets
9,178,923,008/9,480,347,648 bytes. GPU DXGI sampled per-process local peak
983,146,496 bytes (not total-system VRAM or a guaranteed transient peak). One run
per device, process-cold rather than flushed disk caches; no universal speed claim.

Both five-action batches pass requested frame counts, finite transforms/FK, GLB
roundtrip and same-device cold/warm pose equality. GPU constraint errors before
correction: full-pose 1.57 cm mean / 4.01 cm max, right hand 1.45 cm, future path
3.28 cm. Strict 3-cm foot-height diagnostic has zero qualifying samples, hence null
slide metric rather than a zero-slide claim. Same-seed 60-second four-step CPU/GPU
travel differed (22.57/18.87 m): quantization and autoregressive accumulation can
diverge over long sequences. Bounded hidden captures at frames 91/181/1801 show
walk, changed-prompt right-hand wave and long-run poses; intact limbs/floor
alignment were inspected. Captures used only Tide CLI/platform built from the
validated engine checkout D:/Projects/PurrEngine at
28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c. They do not prove all-frame contact quality.

Fresh ZIP install/reinstall, checksum/traversal rejection and model preservation
pass in paths with spaces/quotes. `tests/ardy-runtime.ps1` passes packaged GPU
generation and mixed GPU ARDY / CPU MotionBricks sessions in one process with
Windows-only PATH, then explicit GPU failure and actual auto CPU inference in an
isolated deployment lacking DirectML DLLs. `tests/ardy-inference.ps1 -Device gpu`
records timings/memory and verifies recipes/GLB counts; `ardy-tests ... gpu`
exercises full network/host parity; `ardy-tests --assets DIR` verifies FK/GLB and
repeatability. All inference/timing and preview validation holds the shared named
TideMotionGpuValidation mutex. Outputs/profiles/models stay ignored under .research.

Limits: text remains CPU and dominates cold time/host RAM; only the stated NVIDIA
Windows x64 machine is tested. AMD/Intel, other drivers, Linux/macOS GPU and adapter
selection are unvalidated/unimplemented. Physical GPU device-loss fault injection
was not performed, though the auto restart handler is implemented. DirectML is
optional and CPU remains available. No Kimodo checkpoint/contact cleanup or
MotionBricks authoring behavior changed. Release build targets now include all
default test executables so existing CTest registration has binaries; no additional
CI dispatch was triggered.

ARDY GPU integration completion: rebased onto dev's MotionBricks controls/frame
fix (706665e), native upgrade (b59b270), batched connects (68dd1c5) and optional
owner-local MoMask (adf5567). Kept all of their sources, notices, contracts and
documentation; resolved only additive CMake options/targets and CLI includes.
GPU and CPU reference parity pass again on the integrated binaries. The combined
DirectML-enabled and CPU-only Release trees pass all nine default CTests (MoMask
experiment remains off). These validations/measurements use MOTION_VULKAN=OFF;
Kimodo Vulkan was not revalidated by this thread. DirectML needs no Vulkan SDK.
The upgrade-containing ZIP also passes the offline self-upgrade, preservation,
rollback/locking/refusal script. Its real GPU package smoke and mixed GPU/CPU
sessions pass again. The five-action packaged GPU repeat took 18.173233 seconds,
peak sampled host working set 9,464,115,200 bytes; warm 183-frame generation
0.237261 seconds. This repeat is an integration check, not a second matched CPU
comparison. Weights and generated artifacts remain outside Git.

Final dev integration also preserves 9c418c5's local-install record and 4aeb907's
Kimodo capability work without changing it. Combined the CMake CLI test arguments
so both capability fixtures and ARDY GPU support are passed. A stale developer
source override lacked the new dependency patch; replaced it with private pinned
kimodo/GGML checkouts under each build directory and applied both repository
patches. No shared source cache was modified. The final CPU-only build and the
DirectML build with Kimodo Vulkan enabled (SDK 1.4.363.0) both pass all ten CTests.
CPU and GPU full-network/host reference tests pass after this integration. The
fresh Vulkan-enabled ZIP passes installer and native self-upgrade checks. The
self-upgraded executable, in a path containing spaces, apostrophes and Unicode,
passes GPU inference, mixed GPU ARDY/CPU MotionBricks, strict refusal/actual auto
fallback and all five inference/asset cases with Windows-only PATH for the runtime
smoke. The runtime test also requires all four DirectML/ORT license/notice files.
Five-action wall time 21.475732 s, sampled host peak 9,442,439,168 bytes, GPU local
peak 983,146,496 bytes; this is an integration repeat rather than a new matched
benchmark. Its cold walk GLB is byte-identical to the original GPU benchmark's
asset, so the original bounded visual evidence still applies. The owner's installed
tool/models were not modified in this ARDY thread. No additional CI was dispatched.

### 2026-10-28 — ARDY defaults to the available GPU

Owner clarified that GPU must be preferred without a device flag. CLI and batch
already defaulted to auto; aligned the internal ArdyModel constructor's default
with that contract and made the primary example omit --device. Explicit cpu still
forces CPU, gpu remains strict, and the default auto tries DirectML then CPU.
No model/dependency pins, inference arithmetic or installed tool/model files changed.

Strengthened tests/ardy-runtime.ps1 to omit --device in native CLI generation,
omit device in the batch generation action, and omit --device in the incomplete
deployment fallback case. Actual recipes must report GPU for both normal cases
and CPU without the DirectML runtime; explicit gpu must still fail there.
Windows x64 Release with DirectML and Vulkan enabled passes all ten CTests. The
fresh packaged executable passes those actual default-device checks on the RTX
4080 SUPER, mixed CPU MotionBricks runtime use, license/notice presence and GLB
frame counts, with Windows-only PATH and TIDE_HIDDEN=1. Inference held the named
TideMotionGpuValidation mutex. Outputs remain ignored under .research/ardy.
Existing platform, CPU text and CPU/GPU numerical-difference limits still apply.

## MoMask ordinary CLI and manual weights (2026-10-28)

Owner requested inclusion in Tide Motion with manual weights. The native CPU ORT
backend is now built into the ordinary executable even with
MOTION_MOMASK_EXPERIMENTAL=OFF (that flag only adds the legacy standalone launcher).
Explicit controls: generate --backend momask; inpaint PROMPT --source SOMA.glb
--begin F --end F; models import-momask BUNDLE_DIRECTORY; models check momask.
The normal cache root follows --models/TIDE_MOTION_MODELS/OS defaults, with a
momask subdirectory. Import verifies the exact pinned manifest and each file,
copies to an exclusive staging directory, verifies the copy and renames it.
Existing destination/staging paths are refused. Check/import do not load ORT.
Raw .tar/.pt checkpoints are not accepted as a prepared ONNX bundle. There is no
automatic MoMask download and models install all/defaults remain unchanged.

Standard actions batches accept backend momask generation and inpaint, preflight
all controls/output paths before model loading and permit sources generated by
earlier actions (clip-dependent validation then occurs when produced). Canonical,
case-insensitive Windows collision checks cover the shared batch. RunContext
reuses all five sessions across interleaved generation/mirror/inpaint actions.
Recipes identify backend, exact bundle and unresolved manually supplied rights.
No graph/weights/RNG/representation arithmetic changed. All runtime assets remain
outside Git/releases; source MIT does not establish checkpoint commercial scope.
The earlier precise missing author grant remains unresolved, rather than cleared
by user import. No evaluation motion, SMPL/SMPL-X body assets or user Python is
needed for this skeleton-only native path. Existing model pins/notices remain.

Windows x64 ClangCL Release, default backend build/legacy flag OFF: CPU and
Vulkan-enabled builds each pass all ten CTests in an isolated TEMP/TMP. WSL GCC
13 warnings-as-errors syntax checks pass for main/MoMask/HumanML; splitting
misleading one-line statements fixes the newly default-compiled GCC warnings.
Actual native manual import/check passes on the private 473 MB bundle. A real
standard four-action walk/mirror/idle/inpaint batch loads MoMask once, writes
valid 59-frame GLBs and consumes its earlier idle output. Independent reference
tokens match; exported outside/edit endpoints remain within 1.1920929e-7 after
quaternion sign matching. Walking animation bytes equal the earlier real native
experiment. This CPU-only inference uses no GPU; no GPU benchmark was repeated.
The prior quality/contact and checkpoint-rights limitations still apply; inclusion
does not claim commercially cleared weights or artist-ready results. See
notes/momask-local.md and examples/momask-local.json for supported manual use.

Final integration rebases onto bb25953, retaining the full Kimodo capability
port, ARDY DirectML and its default auto/GPU preference. A private pinned Kimodo/
GGML source copy receives both existing dependency patches; shared source caches
are not modified. The Vulkan generator's stale nested source cache was moved
aside after changing the source override. Windows Vulkan + DirectML Release now
passes all eleven integrated CTests, including refusal of SOMA-77 inpaint sources
(the shared reader otherwise drops their extra tracks). MoMask remains SOMA-30
only; explicit mapped retargeting is unchanged. Import staging creation is atomic
and the runtime package includes the asset-pin JSON documents, containing no assets.

The 72-entry runtime ZIP contains no weights/ONNX/motion data/developer Python,
and retains all three ARDY GPU DLLs/notices. Existing installer and self-upgrade
integration scripts pass, including model preservation, rollback and refusals.
The owner's normal LocalAppData/TideMotion installation was upgraded through the
native local-archive path; its four existing model files' metadata and user PATH
were preserved. Prepared MoMask weights were imported separately to models/momask
and checked. The packaged native standard example, with PATH limited to Windows,
generates two 89-frame clips and edits the idle at [18,66), loading sessions once.
Independent official-tokenizer/full-sampler reference passes again; all exported
outside/edit endpoint components are within 6.5192581e-8 after quaternion sign
matching. Native inspect validates all three assets as SOMA-30, 89 frames, 30 Hz.
PE imports retain Vulkan and CPU ORT. This is a local dev installation, not a
published release or a commercial checkpoint clearance; no GPU inference repeated.
The final installed executable, rebuilt with bb25953's ARDY default preference,
matches the validated checkout build at SHA-256
c422c7d1013f47bb9936e202afd65800287a4de610da24210e69ac06483ca461.
The final native upgrade preserves the imported MoMask files too; installed
help/model verification pass. Existing inference checks were not repeated for
that header-default change, which does not change MoMask arithmetic or graphs.

## Native DART locomotion backend (2026-10-28)

The owner requested implementation after the initial feasibility gate. The
supported bounded feature is now `dart PLAN.json -o OUT.glb`: native CPU DART
continuation and learned waypoint control with walk, run and hop on left leg.
The earlier source-only DART section is historical; full contracts, evidence,
pins and limitations are in [dart-backend.md](dart-backend.md).

The author's public issue 25 explicitly grants Apache-2.0 for code/checkpoints.
The released inference normalizer is used with that model grant after the owner's
clarification; no independent normalizer prohibition was identified. AMASS
dataset/training restrictions are not asserted to prohibit inference-only model
use. More importantly, the separate SMPL dependency has been removed: the native
predicted-joint path obtains its fixed pelvis offset from the normalizer's
mean(joint0)-mean(transl), initializes with the model mean plus native generated
warmup, and reconstructs a fixed reference from generated positions/rotations.
No full SMPL body/calibration extract, meshes, stand.pkl or evaluation motions
are used. No training/fine-tuning, user Python, hosted inference, procedural
motion generator or pose-blending substitute is introduced.

Source pin: zkf1997/DART bb67ae1ed6ce051080468bf15bc6e54a6c3f8417, retaining its
Apache source license and adaptation notices. OpenAI CLIP source/model is MIT,
source revision d05afc436d78f1c48dc0dbf8e5980a9d471f35f6, official ViT-B/32 digest
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af. Every checkpoint,
normalizer, converted tensor and graph URL/size/SHA-256 is pinned in
notes/dart-models.json and native installation. Dependencies remain pinned.
Native install verifies all raw files before reading fixed tensor byte ranges,
converts F16 to F32 and constructs known graphs without deserializing pickle or
executing TorchScript. Converted tensors match the independently exported
372,168,788-byte file's digest. Generated schema source contains no learned
initializer data; no models are committed or packaged. Five source files plus
the converted cache occupy about 875 MB. `all` keeps its previous selection.

The source VAE's integer latent_std buffer must be replaced with the checkpoint's
F32 value after load; plain load_state_dict truncated it to zero. The development
exporter follows upstream's explicit replacement. Four real native graphs agree
with exported PyTorch tensors to 7.03335e-6 maximum absolute error (limit 2e-4).
Development-only tools/dart_export.py and its pinned environment are not shipped
or called by users. Native graph preparation is byte-identical, including a
fresh-cache HTTPS normalizer download and verification.

Plan version 1 has seed, warmupChunks and segments {text,chunks,target?:[x,z]}.
Targets use absolute world metres and advance below 0.3 m pelvis error or at the
budget; misses remain visible in recipes. Two history frames then eight frames
per primitive at 30 Hz, max below 18,000. All plan/batch paths, labels, ranges and
budgets preflight before model loading. A RunContext reuses four CPU sessions and
three CLIP embeddings. Recipes record plan, actual primitive counts/goal errors,
timing, device/threads, sampler, RNG, source revision and all checkpoint digests.
The host uses x0 cosine10 DDIM eta0 CFG5, sampled Gaussian policy actions and
reframed autoregressive history. This is a forward-only goal policy; gradient
inbetweening, arbitrary text and imported-history conditioning are not exposed.

Fixed explicit 22-joint anatomy maps through existing humanoid calibration to
canonical SOMA with fixed bone lengths and unit rotations. Root coordinates are
preserved for independently checkable world goals. No bone identification,
skinning or non-humanoid solver is claimed. Missing twist/end slots retain their
unmapped reference transforms. Frame rate stays 30 Hz; the 20 Hz HML/SMPL-H
variant is not included. Approximate generated reference geometry and the
decoder's redundant joint/rotation inconsistency limit contact quality.

Windows x64 ClangCL 18.1.8 Release and Ubuntu 24.04 x64/GCC 13 (WSL) builds pass
twelve model-free CTests after concurrent integration, including strict DART plan/CLI and late-batch refusal
checks. TEMP/TMP are isolated under this worktree. Linux native graph parity also
passes at 7.03335e-6; a real Linux three-waypoint clip reaches all three targets.
New Linux warnings and one existing v1.2 converter indentation warning were
resolved without changing arithmetic. macOS/ARM64 inference remains untested.

Windows four-thread CPU batch: 24 actions across eight seeds, one verification
and session load, 48/48 waypoint successes (walking turns and walk/run/hop
changes). Independent exported GLB/FK: mean/max goal error 0.1484/0.2981 m,
recipe/root errors agree within 1e-4 m, bone-length variation <2.7e-7 m.
Continuation run segments are faster than walk in 8/8 clips (mean 2.84 vs 1.19
m/s). Support-leg fidelity is imperfect. Max joint step 22.39 degrees, largest
segment-boundary root step 0.1413 m and velocity change 0.6217 m/s. Foot-slide
proxy averages 3.22 cm/frame (max clip mean 5.87); substantial sliding remains.
These are own prompts/model-generated warmup, not upstream dataset metrics.
Observed process time 30.62 s, sampled peak working set 321,949,696 bytes; median
per-action time 1.01 s after load. No isolated GPU performance claim is made.

A 3,074-frame / 102.43 s continuation exports and reimports, 14.19 s generation,
with fixed lengths independently verified; its slide proxy worsens to 6.22
cm/frame. Native hidden Tide preview at 960x540/51 frames succeeds and was
inspected using the privately staged executable built from clean engine checkout
D:/Projects/PurrEngine at 28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c. No engine file
changes. All inference is CPU, so no expensive GPU work bypassed the shared
TideMotionGpuValidation mutex. Existing ARDY, Kimodo, authoring and contact
cleanup implementations/defaults are preserved. No contact quality improvement,
physics feasibility or deterministic match-simulation behavior is claimed.

Final packaging checks: the Windows runtime ZIP installs/reinstalls under an
isolated path with spaces/apostrophes, preserves model data, and refuses checksum
and traversal failures. Its installed executable generates both DART examples
(146 and 290 frames) and strictly reimports their canonical SOMA assets with
PATH restricted to Windows system directories. Both actions share one session
load. Examples, notices and notes/dart-models.json are included; developer Python
and model files are absent. Native graph declarations are readable generated
C++ operators/protobuf fields, not an encoded model file. Fresh raw-only native
preparation matches every pinned graph/tensor hash. Graph construction is a cold
unoptimized installer function to bound C++ compilation time; ONNX inference is
unchanged. Windows and Linux pass the twelve current CTests after preserving the
concurrent ARDY, MotionBricks, selection, Kimodo capabilities and transactional
upgrade changes. Private pinned Kimodo/GGML source copies receive both current
patches; shared dependency trees are never modified. A missing isolated Linux
TMPDIR initially caused test setup failures; creating it and rerunning affected
checks passes. No inference failure was hidden by that environment correction.
Native asset checks independently compare 4,624 60 Hz half-frame samples from
the 24 generated GLBs with double-precision midpoint calculations: maximum
position error 9.53674e-7 m and quaternion dot error 8.62042e-8 on both platforms.
DART remains a 30 Hz model; no 20 Hz checkpoint/resampling support is claimed.

Final integration baseline is 24f17c9, preserving ARDY's available-GPU default,
DirectML packaging, ordinary manual-weight MoMask and every concurrent feature. All twelve CTests pass on both
Windows and Linux; DART tests now compile with warnings as errors as well. Final
packaged Windows smoke: 3 actions, one load, 6/6 goals, 4.20 s wall time; its three
assets are byte-identical to the earlier measured baseline. An intentionally
unreachable [100,100] target with one chunk produces ten valid frames and reports
reached:false / 141.44 m error; no endpoint correction or budget overrun occurs.
Linux TGZ installer/reinstaller, checksum and unsafe-link refusals pass; staged
native generation/reimport succeeds with PATH=/usr/bin:/bin and no loader-path
override. Raw-only native Linux preparation matches every graph/tensor digest.
No additional inference algorithm changes were needed after this integration;
the earlier bounded visual evidence therefore still applies. No installed owner
executable/model cache, engine source, release or additional CI run is changed.


## Parry requests: preview framing, props, turning targets, Mixamo mapping (2026-10-28)

Owner requests from making Parry's sword-duel animations with 0.5.1 (findings in
tide-prototypes notes/animation-pipeline.md). Baseline: origin/dev 01f51a8. No model,
dependency or license change; nothing generated or third-party enters Git.

Preview framing: `body_scene` framed from frame 0, so a get-up starting on its back
got HEIGHT 0.5 (the minimum) and the camera sat inside the body, then showed only
legs. Framing now comes from the whole clip (every second frame and the last):
the top joint, the floor as the median of each frame's lowest joint (a hand pushed
through the floor for a moment does not lift it) snapped to y=0 within -0.3..0.15 m
(a retargeted Synty body lying down sinks its fingers 12-30 cm; no contacts are
solved), and the widest horizontal reach from the motion root. HEIGHT (vertical
extent, camera aim and tiles) and VIEW (camera distance and panel spread, at least
(reach+0.15)/0.63, a panel's half-width) are separate, so a wide lying body pulls
the camera back without lifting its aim. The skeleton drawing uses the same extent.
Measured scene constants (Synty character / mannequin), before -> after: get-up
(owner's h2_getup seed 0) HEIGHT 0.5 / 0.5 -> 1.566 / 1.554, VIEW 2.322 / 1.867,
Synty FLOOR -0.109 -> 0; standing idle 1.5748 -> 1.5753 / 1.5501 -> 1.5628 (VIEW
equal, at most 0.8% zoom); arms-up cheer 1.575 -> 2.189 (its raised arms now fit).
GIFs inspected: the get-up is whole and centred throughout; the idle looks unchanged.

Props: `preview --prop MESH.glb|.fbx --prop-bone BONE [--prop-offset X,Y,Z]
[--prop-rotation X,Y,Z]` adds `asset Mesh<Corner> prop` and draws it in plain steel
with Math.Mul(place, Unscaled(pose.Model(bone)), TRS(offset, Euler degrees)), as
tide-prototypes tools/motion/swordview/preview.tide does; Unscaled normalises the
matrix columns, removing Synty's 0.01. The bone is checked before Tide runs against
every named node of the character file (new node_names: GLB nodes, or FBX nodes as
ufbx/Tide import them), because Synty's Prop_R carries no skin weights and is not in
load_rig_glb's skin rig, yet Tide's skeleton binds it. On the mannequin any SOMA
joint is accepted (decision: attach, not refuse); SOMA RightHand's axes are rest
axes rather than a grip, so a weapon lies along the forearm unless turned. Refused:
--prop with --skeleton, without --prop-bone, a non-mesh extension, unknown bones,
offset/rotation without --prop and malformed triplets. Verified: the Synty sword on
Prop_R with no offset matches the owner's swordview GIF of the same clip frame for
frame (rightcut-soma, 25 frames).

Targets that turn: the owner's turning cut (prompt "turns to the left and slashes",
2 s, idle start/end, --end-at 0,0,60, placement free, right hand pinned at frames
24-31 from rightcut-soma 9-16, seeds 0-3, v1.1, 50 steps, RTX 4080 SUPER under
TideMotionGpuValidation) was regenerated with the baseline binary; it reproduced the
owner's 0.5.1 recipes exactly. Metrics by independent GLB FK (.research turncut.py,
not shipped): head drop 0.65 m, hips 0.40 m (the same cut going straight, which
also has a recovery keyframe at 40: 0.40 / 0.26; the source cut alone drops its
hips 0.21 m and twists them 81 degrees); hand-to-target error in the body's own
frame (hand relative to hips, turned by the hip heading) 9.4 cm mean / 12.8 max.
The drop is at frames 36-44, after the last pin, while heading overshoots to ~105
degrees and returns to 60: not the pinned path fighting the turn.

Two designs were implemented and measured on the same seeds:
- ends: a keyframe/effector at frame f takes t=(f-start)/(end-start) of the end
  pose's placement (yaw turn about its original hips plus the ground move),
  applied before encoding. Head/hips drop 0.64 / 0.38; body-frame hand error 9.5 /
  13.4 cm; swing heading +27 degrees further round (85 vs 58).
- body: a learned guide pass without those targets (as free placement does) put each
  target at the guide hips' ground position and facing at its frame. Drop 0.65 /
  0.40; hand error 9.7 / 13.1 cm. On the straight step-cut it moved the swing's hip
  heading from +55 to -9 degrees: the guide does not twist like the cut, so placing
  targets at its facing erases the move's own hip rotation. Not shipped (also twice
  the sampling).
Adding the recovery keyframe (frame 40 from source frame 24) fixed the kneel in every
mode: world 0.44 / 0.28, ends 0.44 / 0.27, body 0.45 / 0.28; hand error unchanged
(9.5-9.7 cm). Shipped: `--follow world|ends` (batch `follow`), default world
(byte-identical animation payload to the baseline on all four seeds); recipes
record `follow` and per-target `followedTargets` turns. ends needs an end pose and a
keyframe or effector; turns take the shorter way round. A rerun of ends with the
final binary matched the measured output byte for byte. Advice and numbers are in
PROMPTING.md (Pinned swings that turn). Sword GIFs of before / ends / keyframe /
keyframe+ends on the Synty character were inspected: only the keyframed rows stay up.

Mixamo: examples/mixamo.json maps mixamorig: names (Spine/Spine1/Spine2 to SOMA
Spine1/Spine2/Chest, UpLeg/Leg to Leg/Shin). Checked on a real Mixamo rig (IRONVALE
Stylized Lumberjack, owner-bought, 52 bones, local only): Walking, Turn and death
--to-soma and back onto the same FBX; new developer mode `retarget-tests --compare A
B [BONES.json]` compares world positions/rotations of shared (mapped) bones: mapped
bones within 3.0e-7 m and 0.04 degrees on all three. Unmapped fingers return to the
rest pose (SOMA has none): 0.3-0.9 cm mean position difference over all 52 bones.
Lumberjack clips via SOMA onto the Synty character and the round trip on the
lumberjack were viewed as GIFs. Parry's "Standing Melee Attack Downward" is not
mixamorig-named: Mixamo kept the uploaded Synty skeleton, so it takes
synty-polygon.json (round trip exact: 4.9e-7 m, 0.07 degrees) and mixamo.json
refuses it (mapped bone not found).

Also noted by the owner, not acted on: --correction kimodo mixed on planted-foot
slide, 16-28 degree pre-correction errors at their sword idle (consistent with the
documented upstream behaviour above).

Validation: Windows x64 ClangCL 18 Release with Vulkan, all twelve CTests pass
(TEMP/TMP isolated under .research). New contracts: follow_ends geometry (half turn
about the end pose's hips, half move, height and body-local rotations kept, start/end
unmoved, unplaced end is identity, refusal without an end pose); CLI/batch follow
refusals and acceptance through model preflight; preview prop acceptance on an FBX
custom rig and the mannequin, and every refusal; help text. Visual evidence used a
Tide built from engine origin/dev 367ae4da2 in a private detached worktree (the
D:/Projects/PurrEngine checkout at 28b61e4db predates `tide run --record`); the CLI
and MinGW platform were built there and staged under .research, with the unbuilt
MinGW host tools filled from the same tree's release build. No engine source
changed. Linux/macOS builds are left to CI.
## MoMask shared controls and native GPU (2026-10-28)

Owner asked whether backends support the same features and to add feasible
support. Audited the exact pinned MoMask interfaces: text, whole-body RVQ tokens,
residual history/quantizer and padding; no Kimodo spatial constraint inputs or
ARDY autoregressive history interface. Added real application controls instead
of claiming those learned capabilities: --variants 1..64 (seed sequences/session
reuse/contact sheets), --threads 1..1024, --device cpu|auto|gpu, --in-place generation,
--rig soma30|soma77 and matching standard batch fields. MoMask CPU/four threads,
all backend defaults/model pins and manual-weight installation remain unchanged.
In-place is refused for edits. Source SOMA-77 now keeps its existing extra tracks
at every frame and its rig/rest offsets; body edits pass through SOMA-30. All
outside and endpoint poses are restored explicitly. Static presentation fingers
on new SOMA-77 generations are recorded as such, not predicted fingers. Other
custom rigs remain explicit retargeting work. Variant/partial/collision/seed
overflow controls preflight before session creation, including the uint32 edge.

Native Windows x64 DirectML uses the existing pinned/licensed ORT 1.24.4 +
DirectML 1.15.4 distribution already packaged for ARDY. No additional runtime
archive/dependency or model re-export is needed. The optional MoMask GPU TU has
its own C API table; only owned numeric byte vectors/shapes/type IDs cross the
CPU ORT 1.31.1 boundary. It validates float/int64/bool inputs, output shapes and
sizes; existing finite-value/model checks remain. CLIP stays CPU; four motion
graphs use DirectML with CPU auxiliary nodes. Telemetry is disabled. GPU is
strict; auto logs initialization fallback or restarts a GPU-failed current job
from its original seed on CPU before any asset write. Missing-runtime strict
refusal and actual auto CPU inference pass; physical runtime device-loss injection
was not performed. Context sessions reload when device/threads/cache change.
MOTION_MOMASK_DIRECTML is independent but initially follows the ARDY option on
fresh builds. Shared DLL staging/notices preserve ARDY's existing behavior.

GPU validation held TideMotionGpuValidation, used TIDE_HIDDEN=1 and private test
TEMP/TMP. All 12 real motion-network PyTorch fixture cases (4/7/49 tokens) pass;
encoder integer indices match exactly. Max absolute errors: masked 9.78e-6,
residual 2.39e-5, decoder .00155258. GPU bound .002+.002*abs(reference) was fixed
before running; the tighter CPU bound is unchanged. Provider profiles contain
DML events in all four graphs and CPU auxiliary events (not all-operator GPU
execution). Per-process sampled GPU local peak 351,043,584 bytes, not total VRAM.
All eleven model-free CTests and WSL GCC warnings-as-errors syntax checks pass.

A real seven-action batch generates matching CPU/GPU walk tokens, an identical
warm GPU repeat, a SOMA-77 idle/edit, two in-place SOMA-77 variants, then changes
CPU thread count. All eight GLBs have 89 finite local/FK frames. Independent
official-tokenizer/full-sampler reference passes. Model sessions are reused for
all consecutive equal device/threads jobs. Three-second, 18-step generation:
CPU .1364 s, first GPU .4916 s, warm GPU .1469 s, excluding session construction;
single local observations, so no GPU speedup/default change is claimed.
Independent GLB checks, including a privately authored varying index-finger
fixture, verify every extra-joint frame and all outside/end local/FK poses after
native GPU inpaint. Final repeat component error <=5.97e-8 with quaternion sign
matching. No imported evaluation dataset, SMPL/SMPL-X body or user Python is used.

The existing native variant renderer captured a 960x1088 contact sheet with
Tide built from D:/Projects/PurrEngine at 28b61e4db and privately staged here;
the capture uses three hidden frames. Plausible walking poses were inspected.
The optional GIF attempt correctly refuses this engine's missing run --record
feature, retaining all assets/PNG; no GIF success is claimed. MoMask generation
needs no renderer. No Tide engine/CLI source was changed. Existing foot-contact
quality limits remain. See notes/momask-capabilities.md for supported controls,
exact remaining spatial/history research tasks and reproducible native/developer
checks; these are distinct from the unchanged checkpoint commercial-rights gap.

Integrated DART 01f51a8 without changing its source/model/default contracts.
All twelve combined CTests and the staged native Windows-only-PATH runtime smoke
pass, including GPU generate -> mirror -> inpaint with one MoMask load, valid
SOMA-77 reimport and actual auto CPU fallback in a deployment lacking GPU DLLs.
The first smoke exposed a new export offset bug: SOMA-30 hand endpoints have
different parent-local offsets in the 77-joint rig. Removed their inappropriate
translation/rest override on generation, using the existing presentation
converter unchanged; inpaint now always preserves its input rig. Source SOMA-77
extra-track restoration remains explicit and tested. A repeat of all affected
contracts and real packaged inference passes after the fix. No weights changed.
Final rebase also retains 029d645's preview/props/follow/Mixamo work and both CI
fixes. Applied the same size-asserted FARPROC bit copy to MoMask's isolated GPU
entry point for newer Clang compatibility. The combined Vulkan/DirectML build
passes all twelve CTests and the Windows-only-PATH native package smoke again.
No ARDY, DART or Kimodo model arithmetic, weights, defaults or engine code changed.
Final runtime archive contains no model/motion assets or developer Python. Native
local-archive upgrade refreshed the owner's ordinary installation; all twelve
existing model-file metadata records remain unchanged, and private MoMask bundle
verification passes. Installed executable matches the validated Vulkan/DirectML
build, SHA-256 d4b524f0dc0f6b6a701e0ef053e5bdc6d020fda3157243137480dbcaaa526761.

## DART native GPU and inference batching (2026-10-28)

Owner requested GPU acceleration and fuller model utilization. DART now prefers
Windows x64 DirectML with --device auto (default); cpu forces CPU, gpu is strict.
It reuses the existing pinned ARDY DirectML/ORT redistributables and notices in
bin/ardy-directml, adding no SDK/CUDA/Python dependency. MOTION_ARDY_DIRECTML
currently controls both native GPU paths. Separate C API tables isolate GPU ORT
1.24.4 from CPU ORT 1.31.1. Text remains native CPU, cached once. An automatic GPU
initialization failure falls back; an API execution failure restarts the entire
unwritten group from its seeds on CPU. Explicit gpu propagates failures.

Three new opset-17 dynamic-batch graphs reuse all existing tensor offsets; no
checkpoint, weights hash or representation changes. Native model installation
constructs the additional verified graphs; rerun models install dart after upgrade.
Pins are in dart-models.json. Developer-only dart_batch_export.py checks PyTorch
against ORT for batches 1/2/4/8/16. It explicitly transposes decoder latent axes
from [B,1,256] to [1,B,256] and broadcasts the policy's shared standard deviation.
Neither issue could be detected by the previous batch-one exports. Dynamic
graphs have 1439/964/86 denoiser/decoder/policy nodes and share the same 372 MB
tensor file. No model enters Git; developer tools are absent from runtime packages.

Consecutive independent dart actions with equal device/threads/batchSize share
policy and decoder calls and both CFG branches in one 2N denoiser call per step.
batchSize is 1..8, default 8 (--batch-size in internal/standalone arguments).
Every stream retains its own RNG, history, warmup, prompt/target progression,
world transform and metrics; finished streams are padded and ignored. There is
no batching across dependent autoregressive chunks or non-DART actions. All
plans/outputs preflight before inference; a group writes only after all results
exist. Session sets persist across groups, cached by network/static batch size.
Recipes distinguish actual device/accelerator, group wall time, request count,
primitive waves, denoiser calls and sampled process-local GPU memory. A successful
rollout does not imply contacts or exact goals; misses remain explicit.

Windows ClangCL 18 Release/RTX 4080 SUPER, four CPU threads, mutex held throughout
every GPU test and each matched CPU comparison: native PyTorch reference maximum
error CPU 1.04308e-5, GPU 2.19867e-5 (original 2e-4 bound). GPU profiles for each
network at batches 1 and 4 record one fused DML kernel and no CPU-provider kernels.
A DirectML crash when cloning provider session options was reproduced and fixed
by creating/configuring independent options for each cached session.
An intermittent native crash was then observed in a short packaged rollout.
Sessions now use one retained DirectML device and compute queue through DML1,
instead of independent device creation for each network. Twelve independent
GPU continuation processes, repeated full 24-action runs and packaged GPU/
strict-refusal/actual-CPU-fallback checks pass after that change. Exact driver
crash causality was not established; the shared-device path avoids that lifetime
pattern and is the validated implementation. Timings below are the final repeat.

Matched 24-action/eight-seed runs (wall time includes verification/load; single
machine observations, not universal performance guarantees): CPU group 1/8 took
27.746/18.755 s; GPU group 1/8 took 9.065/3.597 s. Every run reached 48/48 goals.
Mean/max goal error stays 0.14838/0.29810 m across modes; independent GLB/FK
checks all 96 assets and recipe roots, with max bone error <2.7e-7 m. Largest
joint step <=22.43 degrees. GPU local memory peaks 224,059,392 / 231,858,176 bytes;
100-ms-sampled host working sets 621,797,376 / 609,529,856 bytes (CPU 320/326 MB).
Foot drift/support-leg limitations remain; acceleration does not fix contacts.

Windows and Ubuntu GCC 13 pass all twelve current CTests with isolated temp paths.
Windows archive installer/reinstaller/refusals pass. The installed executable
runs real default-GPU inference with Windows-only PATH; a private deployment
missing GPU DLLs refuses explicit gpu and really generates on CPU with auto.
tests/dart-runtime.ps1 reproduces it under TideMotionGpuValidation. Models and
generated/reference data remain ignored. No owner installation or engine changes.

Capability audit: released-policy waypoint control and changing supported text
segments use the full native forward host loop. Arbitrary text continuation still
needs native CLIP tokenization/embedding integration and its own adherence tests;
it is not a policy skill extension. Imported SOMA history still needs an explicit,
independently validated inverse representation/calibration, rather than assuming
SOMA and SMPL rest geometry coincide. Those are technical work not delivered here.
Upstream optim_mld.py (noise.requires_grad_, Adam, loss.backward) and
optim_scene_mld.py require gradients through DDIM/decoder/world transforms;
forward-only GPU ONNX calls do not implement inbetweening or scene optimization.
Those need native backward graphs/optimizer plus separately audited scene/body
assets as applicable. No new rights grant or unsupported capability is asserted.

Final integration onto cbff1cc preserves the independent MoMask GPU option,
shared authoring controls, preview/props/follow/Mixamo work and newer-Clang fixes.
The combined Windows DirectML build and Linux CPU build pass all twelve CTests.
The final isolated ZIP installs/reinstalls and passes real default DART GPU,
strict GPU refusal and actual CPU fallback with Windows-only PATH. Models and
the owner's installed tool are untouched. No additional CI dispatch or engine
source changes. GPU timings/memory above use the final shared-device code;
the integrated code changes no inference arithmetic after that measurement.


## MoMask full native authoring controls (2026-10-28)

Owner requested the complete useful tool set and continued native implementation,
including privately supplied weights after the separate rights audit. Implemented
ordinary CLI/batch continuation, append (`extend`), up to 16 changing prompts,
.2..600 s generation, long and disjoint-range inpainting, full-body pose-token
context, mapped pose references, placement/follow/end-at, paths/headings/dense path,
hand/foot effectors, fit/footfall seed search, loops, contact-plan export and sampler
variants. See [momask-capabilities.md](momask-capabilities.md) and the new executed
`tests/momask-authoring.json`; `examples/momask-authoring.json` is a smaller own-prompt
workflow. Existing backends, model assets, dependency revisions and defaults stay
unchanged. No engine/CLI source changes, checkpoint/evaluation-data publication,
model download, hosted inference or user Python dependency.

Every neural window remains <=49 RVQ tokens /196 native 20 Hz samples. Long output
uses bounded observed-prefix token continuation and independent per-window RNG
streams, appending only real decoded frames. Context overlap is 0..4 tokens
(default 1); residual context may optionally stay fixed. Prompt segments and history
produce exact rounded 30 Hz output frame counts, unlike unchanged short-generation
token rounding. `extend` preserves the complete source prefix; source SOMA-77 keeps
every original extra track and holds its last extra pose in the new tail. Output
including prefix is <=18000 frames. New SOMA-77 still has static relaxed extra joints,
not new finger predictions. No automatic rig identification/skinning/nonhumanoid
claim is added.

Continuous long editing uses generated prefix context with native lookahead,
then independently encodes original suffix context in the final window. The
alternative anchored mode retains source poses at internal boundaries. Real
prefix/suffix/pose observations share an explicit heading reference; missing motion
is masked, not interpolated into a placeholder clip. Root boundary alignment,
linear endpoint drift correction and configurable local pose fades are explicit
postprocessing. Outside range and endpoint poses are restored exactly before
serialization. `preserve-root` additionally restores every source root transform
on body-only edits; conflicting pose/path targets refuse. Short single-range
editing retains its existing complete-window model path. Strict outside/prefix
contracts are restored after spatial correction, including source SOMA-77 tracks.

Full-body poses supply observed whole-body RVQ tokens. Exact pose placement/root
warps, path yaw/translation and effector IK use the existing native correction
implementation. No Kimodo learned constraint input or ARDY autoregressive input is
asserted. None leaves anchors approximate and refuses paths/effectors; edit/prefix
boundaries are always restored. Candidate fit ranks pre-correction error, footsteps
filters existing geometric travelling-footfall measurements. Search uses independent
seeds up to64; no matching candidate writes no asset. Loop performs a real second
endpoint-token inference, then explicit in-place/closure correction. It guarantees
tested endpoint transforms, not contact or derivative continuity. Decoded contact
channels drive optional MotionCorrection or qualifying heel-plan export, never
fabricated contacts; no qualifying interval means no plan file. Existing cleanup
work remains separate. Static and existing mapped pose inputs preflight even when
an earlier batch output defers source timeline geometry. Output, recipe and contact
plan collisions are reserved across actions. Equal model/device/thread jobs reuse
sessions; auto GPU execution failure restarts the whole current search on CPU.

New sampling controls are categorical/Gumbel/greedy base selection, residual levels
0..5, independent residual temperature/top-k threshold and unconditional text zeroing.
All recipes retain controls, observations, token masks, actual codes, window prompts,
RNG streams, before/after constraint errors and both requested/smoothed path errors.
A newly exercised logical one-token residual exposed an upstream mask-construction
bug: zeros_like(mask[:,:2]) only creates one prefix at length1. Native residual steps
use one real plus one masked padding slot, then retain the first logits. Independent
PyTorch/ORT/native GPU tests validate this shape; no graph/weight/pin changes.

All-frame quality measurements caught a pre-existing HumanML-to-SOMA roll bug:
Y roll hints were parallel to Y-directed bones, amplifying small RIC/6D decoder
inconsistency into ~119-degree adjacent twists. Representation version2 chooses a
perpendicular raw-bone basis and uses mapped collars for central torso yaw. An
independent tiny-feature-perturbation invariant, shared-reference fragment test,
analytic FK and RIC/normalization roundtrips pass. Default walk model tokens and
root translations remain identical, but saved rotations intentionally change:
maximum adjacent local turn drops to11.54 degrees. Earlier sparse-pose validation
missed that defect; earlier statements of plausible poses did not prove continuous
quality. The fix does not erase model-induced or correction-induced limb turns.

Windows ClangCL18 Release and GCC warnings-as-errors syntax checks pass. All twelve
model-free integrated CTests passed; affected HumanML/MoMask tests pass again after
the final fixes. A real21-action native suite passes on CPU and DirectML GPU, with
own prompts/source clips only. CPU stress adds a600 s /18000-frame asset in56.923 s
native sampling/conversion. Complete CPU stress batch elapsed71.16 s, sampled peak
working set630,239,232 bytes. GPU suite elapsed64.96 s, sampled peak724,979,712 bytes.
Single local timing observations, excluding session construction per asset:
3 s walk CPU .151 s/GPU .342 s; 8 s timeline .825/4.718; 30 s continuation
2.772/16.281; long [180,740) edit1.972/9.809. Dynamic-shape compile and concurrent
machine activity prevent general performance claims; CPU stays the default.

Independent developer-only official-tokenizer/ORT replay matches all native token
choices for sampler modes, reduced/zero residual, unconditional, pose anchors,
preserved residual context, continuation streams, long editing, prompt changes,
one-token generation and both loop passes. An additional12 real1/2/3-token
PyTorch-reference cases pass unchanged CPU bounds1e-4+1e-3*abs(reference) and native
GPU bounds .002+.002*abs(reference). RVQ encoder codes match exactly. GPU max absolute
errors masked/residual1.19e-5, decoder .0001297; small-case GPU-local process memory
peak326,729,728 bytes. Previous4/7/49 cases remain applicable; no neural arithmetic
or asset change affects those shapes. All GPU work holds TideMotionGpuValidation;
all test TEMP/TMP paths are isolated.

Independent GLB tracks/world-FK checks pass every outside/end/root-locked transform,
SOMA-77 extra frame and extension prefix. Largest preserved component error after
quaternion hemisphere matching5.96e-8 CPU /8.94e-8 GPU. A separate authored varying
index-finger fixture passes GPU inpaint and6 s extension checks. A mapped custom
humanoid GLB pose also passes native generation and independent target-FK checks.
Full-body target errors <3.6e-7 m on the main cases, hand target residual .01267 m.
Dense-path max error .04004 m to smoothed points, .05087 m to originally requested
waypoints; both are reported, no exact path claim. Tested loop endpoint transforms
match, root peak speed .6064 m/s. Measured local adjacent turns still reach26.13
continuous-long edit,42.96 extension,68.01 short multi-range and71.32 anchored edit.
Those are real quality limitations, not solved by endpoint equality. Feet may slide
or penetrate; no universal semantic/quality ranking against Kimodo is established.

A 1920x2178 eight-asset contact sheet was captured with three hidden frames and
inspected: walk, kick, keyframes, path, hand target, loop, timeline and continuous
edit sampled poses look plausible. Tide CLI/native platform were rebuilt/staged
from the same engine checkout367ae4da2d31913176ae1e2624e1d78a1204f03c. Full component
staging encountered the existing unbuilt tide-editor target, so the required native
runtime/toolchain component was staged separately from that checkout's mingw-release
build. This is bounded preview validation, not engine installer QA. Sparse images
do not certify every frame. Models/fixtures/captures stay ignored and private.

Checkpoint commercial scope is still unestablished. The code MIT notice does not
supply an author grant for the exact RVQ/masked/residual weights/statistics; dataset
training restrictions alone also do not conclusively decide separate model/output
rights. No new grant is inferred from the owner's private experiment or a mirror
badge. Exact pins/notices and concrete missing-grant questions remain unchanged in
momask.md, momask-private-inputs.json and momask-private-bundle.json. This adds a
usable private manual-weight tool set, not commercially cleared model distribution.

Integrated onto origin/dev f08cb0e, retaining current DART GPU batching, native
advisory review/calibration, equipment export/grip/bow work and all ARDY/Kimodo
contracts. Resolved shared help/batch/output-reservation and appended-note conflicts
by preserving both implementations. Combined Vulkan/DirectML Release build passes
all eighteen CTests, including equipment/review and the expanded MoMask preflight.
No model graph, conversion/sampler arithmetic or neural asset pin changed during
integration. The current source-only evidence and assets were validated before
packaging; no extra CI dispatch or PR is used.

Final packaged native checks also pass: the full21-action authoring suite with
Windows-only PATH, real GPU generate/mirror/inpaint with one reused model set,
strict missing-GPU-runtime refusal and actual auto CPU inference. Independent
exported asset/FK checks remain passing after integration. Native archive upgrade,
reinstall, preservation, locking, rollback and refusal tests pass. The runtime ZIP
contains no models, motion assets or Python files. Included the concurrent equipment
user guide in runtime packaging after finding its CLI link otherwise lacked the
installed document; no equipment code changed. Developer reference scripts remain
outside runtime packages.

Native local-archive upgrade refreshed the owner's ordinary PATH installation at
%LOCALAPPDATA%/TideMotion. All 12 model-file length/timestamp records are unchanged;
manual MoMask bundle verification passes. Installed executable matches the combined
validated build, SHA-256 3eb5dbba031bf8ca1e549daa3baa273d0296fb88eeaea8e544d0d9ab46da7496.
Installed guide/examples contain the new controls; weights and developer tooling
remain separate. This deployment preserves concurrent backends and settings.

A concurrent push rejected the first non-force dev update. Rebased again onto
338c4e1, preserving DART's new native text/source/gradient controls, source/normalizer
pins, notices and graph declarations. Rebuilt the combined Vulkan/DirectML tool;
all eighteen CTests pass again. MoMask neural arithmetic and generated transforms
are unchanged by that integration. A fresh five-action Windows-only-PATH run passes
prompt sequence -> append -> disjoint root-locked edit, path with a60-degree heading
and full-pose context. Independent exported FK/preservation checks pass. Added pose/
mapping SHA provenance and effective placement/follow/path controls to recipes;
path records now also retain requested yaw and independently checked heading error.
No benchmark is rerun for those metadata-only additions or unrelated DART arithmetic.

Latest combined package again passes real native GPU/session reuse, strict refusal
and CPU fallback checks. Final ordinary installation matches the validated binary,
SHA-256 a3e07b7e7541b2bb3d0d85b33b2a662756dc3b35f8f3d9fd8bf6505d045ce67a. All twelve original model length/timestamp records remain unchanged;
six additional Dex2HOI cache files appeared during concurrent work and are retained
untouched. The initial exact-count assertion correctly detected those additions;
verification was corrected to check preservation of the original records without
rejecting unrelated new files. MoMask's exact private bundle verification and new
installed CLI help pass. Package includes only native runtime/docs/notices/examples,
with no model, motion or developer Python payloads. No model rights status changed.
