# DART controls and native optimization

Current capabilities, 2026-10-28. `dart PLAN.json -o CLIP.glb` supports free-text
timelines, source/history continuation, waypoint control, soft pose inbetweening,
sparse joint constraints and joint/SDF scene optimization. Outputs remain ordinary
canonical fixed-length SOMA GLBs at 30 Hz, for explicit humanoid retargeting.
Offline authoring only; existing generators/defaults are preserved.

```powershell
tide-motion models install dart --accept-licenses
tide-motion dart examples/dart-text.json -o timeline.glb --device auto
tide-motion dart examples/dart-goals.json -o goals.glb --device gpu
```

Windows auto prefers DirectML; gpu is strict; auto restarts unwritten work on CPU
after failure. Linux/macOS builds use CPU; macOS inference is not locally tested.
Text stays native CPU. `--threads N` controls CPU work. Consecutive independent
DART batch actions share real policy/CFG/decoder calls, batchSize 1..8 (default 8).
Inputs produced by earlier batch actions split groups. Existing output paths are
protected by the common batch preflight.
Optimization jobs reuse sessions but run individually; chunks and reverse history
remain sequential. No user Python, hosted service, body model or dataset is needed.

## Version-1 plan

| Field | Meaning/default |
| --- | --- |
| segments | 1..64 objects: text, chunks, optional target and guidance |
| text | Any valid UTF-8, <=64 KiB; empty is unconditional |
| chunks | 1..512 eight-frame primitives per segment; total <=2249 |
| target | Optional world [x,z] metres within +/-100 m |
| seed | Unsigned 32-bit, default 1 |
| warmupChunks | 1..32 generated walk primitives, default 8; native history skips warmup |
| history, historyState | Prior GLB recipe JSON/raw state; last (default) or first |
| source, sourceFrame, sourceMap, sourceClip | SOMA GLB/FBX or explicitly mapped humanoid; selected frame defaults last; optional animation name |
| seedMode | history (two poses) or repeat (one repeated pose, zero velocities) |
| guidance | CFG 0..20, default 5; segment override available |
| sampler | ddim (default) or ancestral ddpm |
| timesteps | Increasing unique indices 0..9; default all ten original cosine steps |
| eta | DDIM stochasticity 0..1, default 0; ignored by DDPM |
| noiseScale, zeroNoise | Initial latent scale 0..10 (default 1), or zero Gaussian initialization |
| policyMode | sample (default), mean or none for goals |
| fixFloor, groundHeight | Explicit vertical SOMA floor projection, default false/0 m |
| optimization | Native gradient objectives/Adam, below |

Unknown fields fail before inference. Choose history or source. Paths resolve
relative to the plan. Sources use the existing importer/resampler at 30 Hz. History
needs two poses; repeat accepts one pose and sourceFrame 0. Named FBX/GLB animations
are selectable. sourceMap uses the existing explicit SOMA-to-character mapping;
bones are never automatically identified. Inverse retarget/FK derives the DART
features. Last-frame derivatives extrapolate two poses. Native history avoids
that round trip, preserving 276 features, fixed offsets and world transform.

Output has two initial history frames plus eight per primitive, including repeat
mode. It does not copy preceding source frames or blend transition poses. Recipes
save dartInitialState/dartState, inputs, sampler, actual device, timing and errors.
Compiler normal RNG transforms do not promise PyTorch/cross-platform seed identity.
Floor projection is export-only; repeat fixFloor/groundHeight settings when
continuing native history if the projected seed height must match the prior GLB.

Ordinary targets stop at <0.3 m horizontal pelvis error or budget. The actor was
trained for walk, run, hop on left leg. Other prompts remain available, explicitly
marked policyTrainingSkill:false. policyMode:none does not steer. Text-only segments
use Gaussian noise. noiseScale also scales the actor's latent; zeroNoise applies
only to Gaussian initialization. Actor mean/sample chooses its distribution.
Accepting text does not ensure adherence.

CLIP uses pinned OpenAI BPE, Unicode 14 NFC/lowercase/classes/HTML entities and a
77-token context. No ASCII whitelist. It does not implement ftfy mojibake repair;
unusual malformed numeric HTML references may differ. Sixteen own English/Unicode/
contraction/entity/long-context prompts match the official tokenizer exactly.

## Pose, path and scene optimization

Native analytic ONNX VJPs differentiate denoising, decoder, every history/world
transform and SOMA retarget/FK. Adam changes latent noise and retains the lowest
objective candidate. No finite-difference optimizer, pose blending, IK, endpoint
snapping or procedural replacement is used. Optimization keeps the exact requested
timeline; segment targets become soft endpoint XZ root constraints. No actor prior
is used. A plain ONNX forward call cannot implement this workflow.

```json
{
  "version": 1,
  "history": "start.glb.json",
  "segments": [{"text": "walk forward", "chunks": 12}],
  "optimization": {
    "target": "finish.glb",
    "targetJoints": ["Hips", "LeftFoot", "RightFoot", "LeftHand", "RightHand"],
    "iterations": 120,
    "learningRate": 0.05,
    "constraints": [
      {"frame": 49, "joint": "LeftHand", "position": [0.4, 1.4, 0.5], "weight": 0.5}
    ]
  }
}
```

Positions are canonical Y-up world metres; frames start at zero. target accepts
SOMA/mapped GLB/FBX (targetMap, targetClip, targetFrame, default last), or a native
recipe's final history. targetJoints defaults all; targetWeight defaults 1. Targets
constrain the last generated frame. Arbitrary constraints allow sampled paths and
partial poses; axes:[true,false,true] selects XZ; weight defaults 1. Recipes report
both objective and exported-FK errors, axes and weights.

Default jointSpace:soma optimizes all 30 exported joints, including floor projection.
Unmodeled presentation joints inherit parents; DART has no independent finger/eye/
jaw animation. jointSpace:predicted uses 22 redundant model predictions that can
disagree with fixed-length GLB FK. Default FK is checked against native conversion
on every optimized asset, failing above 0.2 mm. This is skeleton optimization,
not an SMPL-X surface/shape objective.

Options: iterations 0..10000 (default 32), learningRate 1e-7..10 (default .01),
unitGradient/anneal true, jerkWeight/floorWeight zero. Targets use weighted mean
Huber loss. Jerk is maximum third position difference. Floor penalizes feet below
their initial minimum. scene:{file,margin:.03,weight:1} adds mean squared joint
penetration. Supply a version-1 SDF JSON with dimensions:[nx,ny,nz] (2..256 each),
origin:[x,y,z], positive spacing:[dx,dy,dz], and flat finite values in metres,
X fastest, then Y, then Z. Positive means free space. Trilinear sampling clamps
at borders with zero outward derivative. File <=256 MiB. No scene data is acquired.
This does not guarantee surface/object/self collision avoidance, contact or grasping.

Objectives are soft, can compete or be unreachable. The first two seed poses stay
fixed. Recipes save objective history, selected iteration, noise and remaining
errors. checkGradient:true adds four central-difference directional checks. Zero
iterations measures a baseline. Large budgets can take substantial time.

## Assets, operators and validation

The exact Apache checkpoint/normalizer and MIT CLIP rights investigation in
dart-backend.md remains applicable; dataset access does not replace checkpoint
inference rights. No SMPL-X/H model, rest-body extract, upstream stand.pkl, motion
dataset, training or author contact is used. Model/motion files stay outside Git.
dart-models.json pins all six downloads (now including BPE), thirteen generated
schemas, shared 372168788-byte tensor file, revisions, URLs, hashes and licenses.
Rerun models install dart after upgrade. Installation/inference never executes
Python, TorchScript or a pickle interpreter. All learned initializers are external.

Six new standard-opset-17 graphs provide decoder/denoiser VJP, history/world forward
and VJP, SOMA FK forward and VJP: 1742/2618/546/1618/2364/4474 nodes. Native Adam and
reverse sampler/history need no ONNX training runtime. Developer-only
dart_grad_export.py uses pinned PyTorch 2.7.1+cpu at e2d141dbde55c2a4370fac5165b0561b6af4798b,
verified source/checkpoints and the existing restricted loader. New notices cover
PyTorch-derived operations, CPython/Unicode tables; CLIP BPE remains MIT. Existing
ufbx supplies inflate, adding no runtime library. No new learned weights are used.

Windows ClangCL 18/RTX 4080 SUPER and Ubuntu GCC 13 pass twelve CTests. Native
CPU/GPU gradient references pass abs+relative 2e-4 bounds; max 2.494e-4/2.561e-4 for
large denoiser gradients, <=1.526e-5 other learned VJPs, <=7.63e-6 transforms.
Whole-chain checks cover multi-chunk canonical/predicted joints, stochastic DDIM,
DDPM, native history, floor, jerk and SDF. Not full upstream body-model parity.

Eleven own-prompt GPU assets use one session set: wave/squat/jump, history/source/
repeat, actor mean/untrained text, pose targets and regularizers. Two 50-frame
inbetweens (six chunks, 120 iterations, rate .05) put all 30 endpoint joints within
8.59/9.65 mm. Objectives .021618/.008068 become 2.873e-6/3.253e-6 in 30.55/30.02 s.
Targets/seeds come from this same model. Peak host 1124900864 B (100 ms sampling),
GPU 680607744 B. Scene/sparse targets are numerical checks, not a success benchmark.
A short 18-frame goal still misses by .662 m; constraints are not guaranteed.

Independent FK bone errors <2.5e-7 m; 723 half-frame resamples have position error
<2.4e-7 m, quaternion dot error <7.3e-8. Native objective/FK agreement <1 micrometre;
earlier source/history round trips <.6 micrometre. Pose targets retain 3.52/3.55
cm/frame foot-slide proxies; no cleanup improvement claimed. Hidden 960x540 native
mannequin capture inspected at --frames 51, engine 367ae4da2d31913176ae1e2624e1d78a1204f03c
built from D:/Projects/PurrEngine; engine source untouched.

Reproduce with tests/dart-controls.ps1, dart-inference.ps1, dart-runtime.ps1 and
optional dart-tests token/gradient-reference modes. The final combined Windows/Linux
build passes eighteen CTests, including concurrent tools. implementation.md records
packaging and integration checks. DART adds eight-frame action changes, its actor
and latent optimization. ARDY's SOMA KimodoRep flow/history and text encoder remain
independent. Training, alternate unvalidated checkpoints, body shape/mesh export,
automatic rigging/skinning, non-humanoid retargeting and simulation are not delivered.
