# Dex2HOI manual weights and native experiment

Dex2HOI is an experimental, manual-weight CPU backend for joint human/object
generation. It runs CLIP, the learned BPS geometry encoder and the learned motion
network through the existing pinned ONNX Runtime. Installed inference does not
run Python, contact a service or download checkpoints. Kimodo remains the default.

The owner authorized privately obtaining and testing generation weights while
artifact terms remain unresolved. This changes the local/manual integration
policy, not the upstream grant. The pinned repository has no top-level license;
licenses in vendored subdirectories do not license all Dex2HOI code or checkpoints.
No weights, research motion, upstream body model or new upstream implementation
are distributed with Tide. Native host/sampling code is independently written.
The developer converter imports a separately obtained pinned source checkout.
Preserve supplied notices with any private bundle. See the historical
[release audit](dex2hoi-release-audit.md).

## Install your prepared weights

```powershell
tide-motion models import-dex2hoi C:/private/dex2hoi/bundle
tide-motion models check dex2hoi
tide-motion dex2hoi generate --input C:/own/history.json `
  --prompt "a person lifts the object with both hands" -o C:/own/lift.glb `
  --seed 7 --threads 4
```

`--models CACHE` on import/check selects another cache. Generation can select it
with `--bundle CACHE/dex2hoi`; otherwise it uses the normal local model cache.
Import refuses an existing destination, incompatible manifests, duplicate JSON
keys, changed sizes/hashes, symlinks/reparse points and unsafe filenames. It
stages and verifies every listed file before exclusive publication. Verification
establishes file integrity/configuration, not rights or motion quality. An
existing prepared bundle can be inspected without inference using
`dex2hoi check --bundle DIRECTORY`.

Raw PyTorch `.pt` files are not runtime inputs. A prepared bundle contains
`bundle.json`, `denoiser.onnx`, `object.onnx`, `clip.onnx`, `tokenizer.json` and
`normalization.json`, plus optional explicitly pinned flat files such as notices.
The version-1 manifest records each file's size/SHA-256, upstream revision,
architecture, diffusion configuration, checkpoint/statistic origin pins and
`licensingStatus: "user-supplied-unverified"`. It supports compatible user
conversions rather than one hardcoded checkpoint digest. ONNX graphs must obey
the exact static native tensor contract. Import is a file/manifest check;
generation additionally validates graph interfaces when opening sessions.

## Own history and explicit rig binding

Generation currently produces one window: ten supplied history frames followed
by sixty predicted frames at 30 Hz (70 total). These are the selected upstream
inference settings, not a claim that the missing training `args.json` has been
recovered. Longer autoregression and GRAB are not integrated.

The version-1 history manifest has these fields:

```json
{
  "version": 1,
  "history": "own-ten-frame-prefix.glb",
  "bones": ["65 distinct explicit joint names in checkpoint order"],
  "objects": [
    {
      "name": "sword",
      "mesh": "own-rigid-sword.glb",
      "present": true,
      "exportTrack": true,
      "points": [[0.0, 0.0, 0.0]],
      "poses": [{"position": [0.0, 1.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}]
    },
    {
      "name": "shield",
      "mesh": "own-rigid-shield.glb",
      "present": true,
      "exportTrack": true,
      "points": [[0.0, 0.0, 0.0]],
      "poses": [{"position": [0.0, 1.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]}]
    }
  ]
}
```

The example abbreviates arrays: **65** bone names, **128** local surface xyz
samples per slot and **10** world transforms per slot are mandatory. Units are
metres, Y-up; rotations are unit XYZW quaternions. File paths resolve against the
manifest. Mesh is optional for a transform-only track. Static rigid prop meshes
follow [equipment export restrictions](equipment-authoring.md); the character
mesh is not embedded. Supply points in the same local origin, axes and metric
scale as the prop mesh/poses. The BPS graph internally mean-centres/scales its
conditioning cloud; host fingertip contact encoding uses the metric samples.

The explicit checkpoint order is hips; spine/spine1/spine2; neck/head/head-end;
left shoulder/upper arm/forearm/wrist; five four-bone fingers in thumb/index/
middle/ring/pinky order; corresponding right arm/fingers; left thigh/shin/foot/
toe/toe-end; corresponding right leg. Slots 0, 10 and 34 are hips and the two
wrists. No joint names are inferred. Scales must be baked to one and nonroot
translations must remain at rest. Unmapped helper joints must remain at rest.

**Rest bases matter.** Learned rotations are Blender-style pose deltas, not
ordinary local rotations. The supplied history rig defines those rest bases.
Native encoding/decoding accounts for them, including the rest-stripped wrist
frames used by the object representation. A name mapping onto SOMA-77 or another
humanoid does not establish compatibility with the checkpoint's trained bone
frames or proportions. This is an experimental explicit-binding adapter, not
automatic retargeting. Tide's normal SOMA/offline mapped-humanoid workflow remains
available; there is no automatic bone identification, skinning or nonhuman rig
support. A compatible owner rig/reference is still needed for useful quality tests.

For one prop, the pinned HUMOTO loader duplicates the object into both slots.
To reproduce that conditioning, supply identical points/poses in both entries
and set the second `exportTrack` false. A masked second slot is accepted but
is a different experiment. Two props occupy the two explicit slots.

The GLB contains the supplied body skeleton and independently animated world
object tracks, with supplied rigid meshes when present. The ten source body and
object transforms are preserved exactly before export. Finger rotations for
mapped finger joints come from the network. Extras hold rest after the prefix.
There is no procedural replacement for learned frames, pose blend, forced grip,
weapon collision solver, automatic grasp or automatic bow/string reconstruction.
Existing equipment authoring can inspect/correct explicitly specified grips and
author bow release as a separate offline step. Animated equipment GLBs contain
extra object nodes; select the intended skeleton explicitly when reusing them.

## Sampling and reproducibility

Default sampling is upstream cosine diffusion, all 500 timesteps, x0 prediction,
DDIM eta=0, text-only CFG 2.5. `--guidance 0..10` changes CFG; `--steps 2..500`
explicitly experiments with rounded evenly spaced original timestep indices.
Fewer steps have not been certified for quality. The recipe records the exact
indices, model/input/mesh hashes, token IDs, device/runtime, seed and timing.
Native RNG is `splitmix64-open53-boxmuller-v1`; the same numeric Torch seed does
not imply identical random tensors. This is offline model arithmetic, not Tide
deterministic match simulation. CPU thread count is 1..64, default four.

Native CLIP currently accepts 1..4096 ASCII bytes without HTML entities or special
token syntax, matching the existing MoMask tokenizer scope. Unicode/ftfy parity
is not claimed. The denoiser supplies object x0 conditioning but no separate
root x0 token, matching the pinned HUMOTO loader. Root x0 is retained for host
decoding and the initial root transform remains a network condition.

## Developer conversion and validation

Source: `cpratikaki/Dex2HOI` at
`a92aca6b48b7a0c9a45a1465aa984213d3c8ceda`. Developer imports are checked against
[source-lock.json](../tools/dex2hoi/source-lock.json); package plotting and unused
PointNet imports are bypassed. The actual BPS branch executes unmodified source.
The checkpoint's `model_avg` loads strictly: 429 features, two 18D objects,
512-wide latent state, eight decoder/cross-attention layers, 1024-wide input/output
MLPs and three-head transformer fusion. BPS has 512 checkpoint-stored basis points.
Do not substitute a PointNet checkpoint or invent missing configuration fields.

Use isolated developer Python 3.11 with
[pinned packages](../tools/dex2hoi/requirements-dev.txt) and the PyTorch CPU index.
The converter takes local inputs only:

```powershell
python tools/dex2hoi/export.py --source C:/private/Dex2HOI `
  --weights C:/private/humoto --clip-bundle C:/private/validated-momask-bundle `
  --output C:/private/new-export
```

Weights must provide `model.pt`, `mean.pt`, `std.pt`, `eps.pt`. The currently
supported CLIP source is the existing canonical ViT-B/32 MoMask native export;
the converter pins both its graph and tokenizer metadata hashes. It verifies
upstream source, uses safe `weights_only=True` checkpoint loading, preserves BPS
weights/basis, exports inline ONNX and checks real PyTorch/ORT outputs. Converted
graphs and tensor fixtures are private model assets. No compiler, Torch, PyG,
Blender or Python is needed by the installed native inference executable.

Independent native CTests cover host representation, normalization, explicit
bases/FK, contact variants, file import integrity and sampler equations. Optional
real-checkpoint parity after conversion:

```powershell
build/Release/dex2hoi-runtime-tests.exe C:/private/new-export/bundle C:/private/new-export/fixtures
```

Executed Windows x64 ClangCL 18.1.8 Release validation against the real HUMOTO EMA
weights: three native denoiser cases match Torch within max `1.48e-5`, mean
`6.22e-7`; cached-object decomposition is exact in Torch. Host checks against the
pinned upstream on own synthetic rigs give max object translation `7.15e-7 m`,
FK position `6.56e-7 m`, FK rotation matrix `1.43e-6`, initial conditioning
`1.07e-6`. These numbers measure numerical agreement, not sword/shield contact
quality, first-person composition or anatomy. No speedup is claimed.
