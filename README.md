# Tide Motion

Native, offline animation authoring using Kimodo for text-to-motion and
MotionBricks for transitions between clips. It is a separate tool and repository;
Tide is an optional preview renderer. No Tide CLI or engine changes are required.

The models use canonical SOMA-30 human animations. A reusable bone-mapping file
retargets between SOMA and your humanoid rig, offline and without another model.
Outputs are self-contained GLB files containing an animated skeleton, without a
mesh. They drive the character mesh you already have. Automatic bone assignment,
rigging, skinning and non-humanoid retargeting are not implemented.

## Install

Download an archive, its `.sha256` sidecar and the installer from
[Releases](https://github.com/Tide-Engine/tide-motion/releases/latest).

| Platform | Archive | Requirements |
|---|---|---|
| Windows x64 | `windows-x64.zip` | AVX2 CPU, Windows 10/11; uses a Vulkan 1.2 GPU when present |
| Linux x64 | `linux-x64.tar.gz` | AVX2 CPU, Ubuntu 24.04 or compatible glibc/libstdc++ |
| Linux ARM64 | `linux-arm64.tar.gz` | Ubuntu 24.04 or compatible glibc/libstdc++ |
| macOS Apple Silicon | `macos-arm64.tar.gz` | macOS 15 or newer |

Archive names start with `tide-motion-<version>-`. Intel Mac and Windows ARM64
packages are not currently provided. macOS builds are unsigned and unnotarized;
macOS may require approval in Privacy & Security for a downloaded executable.

Windows:

```powershell
./install.ps1 -Archive ./tide-motion-0.5.1-windows-x64.zip -AddToPath
```

Reopen the terminal after adding PATH. `-Root <directory>` selects another
installation. Linux/macOS (select your platform's archive):

```sh
sh install.sh ./tide-motion-0.5.1-linux-x64.tar.gz
export PATH="$HOME/.local/share/tide-motion/bin:$PATH"
```

Persist that PATH entry in your shell profile. A second argument to `install.sh`
selects another installation directory. Neither installer needs administrator
access; both verify the archive and test the new binary before replacing it.

The archive includes the native executable, ONNX Runtime and, on Windows,
required Microsoft runtime files. No Python, PyTorch, Conda, Docker or compiler is
needed to use a prebuilt package. Model downloading uses the OS `curl` executable,
as Tide does. Models are separate from the executable and survive reinstalling it.

Read `licenses/NVIDIA-KIMODO-MODEL.txt`, `licenses/NVIDIA-OPEN-MODEL.txt` and
`licenses/META-LLAMA-3.txt`, then download the selected weights:

```sh
tide-motion models install all --accept-licenses
```

Both backends total about 10 GB: Kimodo motion weights 1.13 GB, its Q8 text
encoder 8.14 GB, tokenizer 7.3 MB, and MotionBricks 758 MB. Install just `bricks`
when connecting existing SOMA clips. Downloads resume and verify pinned SHA-256
hashes before files become usable. Generation works offline once they are present.
Allow at least 12 GB of available RAM for CPU text generation.

The Windows package generates with Kimodo on the first Vulkan 1.2 GPU when its
driver is installed, and on the CPU otherwise; it says which when generating.
On an RTX 4080 SUPER a 2-second, 20-step clip took 35 seconds instead of 13
minutes on its CPU. `--device cpu` (or `KIMODO_BACKEND=cpu`) forces the CPU,
`--device gpu` fails rather than falling back, and `--threads N` caps the CPU
threads. MotionBricks (`connect`) always runs on the CPU (`--threads` applies
there too). Other platforms' packages are CPU-only for now.

`--models DIR` or `TIDE_MOTION_MODELS` selects the model cache. Otherwise it is
`%LOCALAPPDATA%/TideMotion/models` on Windows, and
`$XDG_CACHE_HOME/tide-motion/models` or `~/.cache/tide-motion/models` on Unix.

## Generate

```sh
tide-motion generate "A person walks forward slowly" --duration 3 --seed 4 -o walk.glb
tide-motion generate "A person stands still and waves" --duration 3 --seed 7 -o wave.glb
```

Choose `--seed` to make candidates and `--steps` to control diffusion work (default
50). Duration is 0.2–10 seconds. The seed reproduces the backend's random draws;
identical output across different hardware or runtime versions is not promised.
Save the resulting assets rather than regenerating them during game builds.
`tide-motion help generate` (or `generate --help`) lists every option.

| Option | Does |
|---|---|
| `--variants N` | Makes N clips (1–64) from seeds `--seed`, `--seed`+1, …, keeping the model loaded, and a contact sheet |
| `--variants-gif` | Also a GIF of every variant playing at once |
| `--guidance W` | How strongly the prompt steers the motion, 0–10 (default 2; see below) |
| `--constraint-guidance W` | How strongly poses and paths steer it, 0–10 (default 2) |
| `--correction-frames N` | Frames each side the pose correction fades over, 0–60 (default 10; 0 sets only the frame) |
| `--in-place` | Removes horizontal root travel, keeping height, facing and sway |
| `--device auto\|gpu\|cpu` | Where Kimodo runs (default auto: the GPU, falling back to the CPU) |
| `--threads N` | CPU threads, 1–1024 (default: all) |
| `--tide PATH` | The Tide that draws the contact sheet (default: `tide` on PATH) |

### Variants and the contact sheet

```sh
tide-motion generate "A person kicks forward with the right leg" --duration 2.5 \
  --start-pose idle.glb --end-pose idle.glb@0 --variants 8 -o kick.glb
```

This writes `kick.seed0.glb` … `kick.seed7.glb`, each with its recipe, and
`kick.variants.png`: one labelled cell per seed showing six moments of the clip
(its first and last frames and its strongest movements, with their times) in
right profile above and facing the camera below, on the mannequin (left limbs
blue, right orange). A kick that didn't happen is obvious at a glance. To
remake one exactly, run the same command with `--seed 5` and without
`--variants`. Names always carry the seed (`-o kick.glb --seed 20 --variants 4`
gives seeds 20–23). `--variants-gif` adds `kick.variants.gif`, every variant
playing at once. The sheet uses an installed Tide, as `preview` does; without
one the clips are still written and the sheet is reported as skipped.

### Mirror

```sh
tide-motion mirror kick.seed5.glb -o kick-left.glb
```

Swaps left and right in a SOMA-30 clip and reflects it across the character's
sagittal plane at its first frame (through its root, upward and along its
facing), so a right kick becomes a left kick that starts in the same place,
facing the same way. No model is loaded. SOMA's rest skeleton is symmetric to
within a fraction of a millimetre, so mirroring twice returns the clip to within
a few millimetres.

### In place

`--in-place` removes the root's horizontal travel for clips that gameplay moves:
the travel is the hips' ground path averaged over a second, subtracted from
every frame, so frame 0 stays put and height, facing and the body's sway remain
(a 3-second walk that travelled 4.85 m stayed within 11 cm). It is applied last;
constraint errors in the recipe are measured before it, and poses after frame 0
move with the travel removed. It refuses `--path`, whose only purpose is travel.

### Getting a good result

[PROMPTING.md](PROMPTING.md) covers what to write and how to nudge a wrong
result, with the measurements behind it. In short: one movement per prompt,
plainly worded, naming the side and direction; 1.5–3 seconds per move, chained
with `--then`; ask for the change ("a standing person crouches down"), not the
state; leave out props and characters; then `--variants 8` and pick. If the move
still doesn't appear, reword, raise `--guidance` to 3–4, or pin its key frame
with `--keyframe` or `--effector` taken from your best variant.

### Prompt sequences

```sh
tide-motion generate "A person walks forward" --duration 2 \
  --then "A person turns around" --duration 1.5 \
  --then "A person waves with the right hand" --duration 2 --seed 4 -o sequence.glb
```

`--then "PROMPT"` adds a segment (up to 16 prompts); a `--duration` applies to the
prompt before it (default 3 seconds each). Kimodo generates each segment from the
end of the previous one, as upstream's multi-prompt mode does, and blends the
`--transition` overlapping frames (default 5, 1–60, shorter than each later
segment). The clip lasts the sum of the durations.

### Poses, keyframes and paths

Constraints come from SOMA-30 clips (generated ones, or your own animation after
`retarget --to-soma`). `FILE.glb@F` picks frame `F` of a clip; negative counts from
its end (`-1` is the last), and `first`/`last` also work.

```sh
# Continue from a clip: starts in guard.glb's last frame (the default for a start pose).
tide-motion generate "A person swings a sword horizontally from right to left with both hands" \
  --duration 2 --start-pose guard.glb -o swing.glb
# Lead into a clip: ends in guard.glb's first frame (the default for an end pose).
tide-motion generate "A person walks forward and raises a sword into a guard stance" \
  --duration 3 --end-pose guard.glb -o walk-to-guard.glb
# Swing and return to the guard, in place.
tide-motion generate "A person swings a sword and returns to a guard stance" --duration 2.5 \
  --start-pose guard.glb --end-pose guard.glb@first -o swing-return.glb
# Hold a mid-swing pose at output frame 60, and a right-hand target at frame 40.
tide-motion generate "A person walks forward and swings a sword" --duration 3 \
  --keyframe 60 side-cut.glb@30 --effector right-hand 40 side-cut.glb@30 -o keyed.glb
# Walk through ground points (metres), ending at frame -1 facing -X (yaw -90).
tide-motion generate "A person walks" --duration 4 \
  --path 0,0@0 --path 1.5,1.5@60 --path 0,3,-90@-1 -o path.glb
```

| Option | Holds |
|---|---|
| `--start-pose FILE.glb[@F]` | Output frame 0 in frame `F` (default: the clip's last) |
| `--end-pose FILE.glb[@F]` | The last output frame in frame `F` (default: the clip's first) |
| `--keyframe FRAME FILE.glb[@F]` | Output `FRAME` in frame `F` (default first); repeatable |
| `--effector PARTS FRAME FILE.glb[@F]` | Only the hands/feet in `PARTS` (`left-hand,right-hand,left-foot,right-foot`) |
| `--path X,Z[,YAW]@FRAME` | The root over ground point `X,Z`, optionally facing `YAW` degrees (0 faces +Z, 90 faces +X) |

Output frames are 0-based at 30 fps; negative frames count from the end. Constraints
work with `--then` sequences too; each pose applies at its frame of the whole clip.
The encoding follows upstream Kimodo: a full-body pose constrains the root
position, height and heading, every joint's root-relative position, and the
hand and foot rotations; an effector constrains its chain's positions and base
rotation with the body's root; a path point constrains the smoothed root.

**Placement.** A start pose (or a path point at frame 0) fixes where the output
begins, in that clip's world, so the result continues seamlessly from it. Later
poses are then pinned to their positions in their files too (`world`), which suits
"swing and come back to guard". Without a start pose, Kimodo begins at the origin,
so the default (`--placement auto`) lets later poses float: their body, height and
facing are kept and the model chooses where on the ground they happen, then the
whole clip is moved so its last pose lands where it is in its file; the
`walk-to-guard` example walks and ends exactly where `guard.glb` begins. Force
either behaviour with `--placement world` or `--placement free`. A pinned pose
that contradicts the prompt is reached less closely (an end pose at the start
point of "walk forward" was missed by 10 cm before correction).

**Accuracy.** Kimodo reaches poses closely but not exactly. Generation prints the
model's error at each constraint, and the recipe stores it; on an RTX 4080 SUPER
full-body poses typically landed within 1–2 cm on average (4–5 cm at the worst
joint) and 3–8 degrees, effectors within about 5 cm and path points within
3–10 cm. Full-body poses are then made exact: the residual is faded out over up
to 10 frames either side (`--correction-frames N`; 0 sets only the constrained
frame, which can pop), ending before any neighbouring constraint, so clips join
without a pop. `--no-correction` keeps the model's poses unchanged. Effector
targets and path points are not corrected.

## Connect clips

```sh
tide-motion connect walk.glb wave.glb --offset 1.5 0 0 -o bridge.glb
tide-motion connect walk.glb wave.glb --offset 1.5 0 0 --joined -o sequence.glb
```

MotionBricks receives the last four source frames and first four target frames,
sampled at 30 Hz. It predicts the intermediate movement. Native endpoint
correction preserves the source's final pose and target's first pose exactly.
`--joined` includes both surrounding clips and removes duplicate seam frames.

**Target placement is explicit.** Input root motion remains in its authored world
placement. `--offset X Y Z` moves the complete target clip before generation; it
does not turn the target or choose a path. A target left behind the source can
legitimately generate a turn back toward it. Inspect the motion before using it.
No foot-contact cleanup, obstacle avoidance or physical validation is guaranteed.

The model normally chooses 8–14 tokens. `--tokens 6..16` fixes the token count.
After trimming boundary frames there are `4*tokens - 6` frames at 30 Hz. Output
duration is `(frames - 1)/30` seconds.

MotionBricks inputs must use the canonical SOMA names, hierarchy and proportions, with no
scale tracks, matrix nodes or animated extra nodes. LINEAR and STEP GLB tracks
are supported; bake cubic tracks before import. A GLB with several animations
requires `--from-clip NAME` or `--to-clip NAME` (`--clip NAME` for inspect/preview).

## Retarget a humanoid

List your character's bones, copy [the example mapping](examples/humanoid.json),
and replace its values with your rig's exact bone names:

```sh
tide-motion inspect character.glb --skeleton
tide-motion retarget sequence.glb --target character.glb --map humanoid.json -o character-sequence.glb
```

Characters and source clips can be FBX as well as GLB (`inspect --skeleton`,
`retarget --target`, `retarget --to-soma`, `preview --character`). FBX is read
with the same ufbx revision and options as Tide's own importer, so axes and units
become Y-up metres the way a Tide game sees the file (a centimetre rig keeps a
0.01 scale on its top bone) and bone names match what the game binds. A Synty
POLYGON character works with [examples/synty-polygon.json](examples/synty-polygon.json):

```sh
tide-motion inspect PolygonSyntyCharacter.fbx --skeleton
tide-motion retarget swing.glb --target PolygonSyntyCharacter.fbx --map examples/synty-polygon.json -o swing-synty.glb
tide-motion retarget A_Idle_Base_Sword.fbx --to-soma --map examples/synty-polygon.json -o idle-soma.glb
```

The mapping's keys are SOMA's anatomical names; values name your character's
bones. The example uses illustrative pelvis/spine/arm/leg names, not automatic
rig detection. Required keys are `Hips`, `Chest`, `Head`, and each side's
`Arm`, `ForeArm`, `Hand`, `Leg`, `Shin`, `Foot` with `Left`/`Right` prefixes.
Other SOMA bones are optional; map toes and shoulders when the rig has them.
`inspect --skeleton` lists parent indices, local positions/rotations and scales.
Bone names and ancestor names must be unique and nonempty. Mapped ancestors must
remain in anatomical order; extra unmapped helper/twist bones are allowed.

The tool builds a matching reference pose by aiming mapped segments along their
SOMA reference directions, including A-pose/T-pose differences. World-space
rotation changes are transferred to the target's bone axes, preserving its local
bone lengths. Unmapped bones retain their calibrated local reference transform
and inherit parent movement; twist is not automatically distributed among helpers.
Optional `referenceRotations` maps character bone names to local `[x,y,z,w]`
quaternions, replacing automatic aiming for those bones when manual calibration
is needed. These values describe the matching reference pose, not an additive
animation rotation.

`rootScale: "auto"` scales root travel by the average leg-length ratio. A positive
number replaces that ratio (in world units); the reverse operation uses its
reciprocal. Hip separation establishes the character's heading. `headingDegrees`
adds a yaw correction, and `groundOffset` adds a vertical offset in character
world units. Reference feet/toes establish ground height; this is not contact IK.
Only Hips translation is transferred; non-root translation animation is reported
and omitted so target bone lengths stay fixed.

To use existing character clips with MotionBricks, convert them to SOMA first:

```sh
tide-motion retarget existing.glb --to-soma --map humanoid.json -o existing-soma.glb
```

Connect on SOMA, then retarget the joined sequence back to the character. Both
directions use the same mapping, preserve frame count/timing, and need no model
files. `--clip NAME` selects the source animation when there is more than one.
The exported GLB contains the target skeleton and animation, without mesh,
materials or textures. In Tide, import it as `Animation` and sample into your
original character's default pose.

Retarget input is self-contained GLB, sampled at 30 Hz with LINEAR/STEP tracks.
Static positive uniform scale, including centimetre armature wrappers and affine
matrices without shear/reflection, is supported. Constant scale tracks are also
accepted on custom rigs. Changing scale, nonuniform/negative scale, morph tracks
and cubic interpolation require an appropriate export/bake first. This is a
humanoid rotation/root-motion retargeter, not foot planting, obstacle avoidance
or a guarantee of a perfect result on every character.

## Inspect and preview

```sh
tide-motion inspect sequence.glb
tide-motion preview sequence.glb --gif sequence.gif
tide-motion preview swing.glb --character fighter.fbx --map examples/synty-polygon.json \
  --texture fighter.png --gif swing-fighter.gif
tide-motion preview fighter-swing.glb --character fighter.fbx --capture swing.png --frames 40
tide-motion preview sequence.glb --skeleton --tide /path/to/tide --capture preview.png --frames 90
```

Preview makes a temporary Tide project and plays the clip lit, through one camera
that shows it twice: facing the camera on the left and from its right side on the
right, turned by the clip's first-frame heading. Each copy is centred on the motion
root over a tiled floor that slides as the root travels, so root motion stays
visible. It uses a separately installed Tide; no engine is bundled.

- **Mannequin** (default for SOMA-30 clips, `--mannequin`): a body Tide Motion
  builds itself, a capsule along each SOMA bone rigidly skinned to it, with a nose
  to show the head's facing. Left limbs are blue and right limbs orange, so a
  crossed or mirrored limb is obvious. No third-party art is involved.
- **Your character** (`--character MODEL.glb|.fbx`): with `--map BONES.json`, a SOMA
  clip is retargeted onto the character first, exactly as `retarget --target`
  does; without it, the clip must already animate that character's rig (an output
  of `retarget`, or one of the character's own clips). Tide imports the model and
  plays the clip on it with its documented `SkinnedMesh`/`Animation` assets.
  `--texture IMAGE` paints it; otherwise it is plain clay (vertex colours are
  ignored, as FBX packs often carry unrelated ones).
- **Lines** (`--skeleton`, and the default for other rigs without `--character`):
  the original front/side bone drawing.

Captures use Tide's ordinary bounded capture mode. `--gif OUT.gif` records the
clip exactly once, from its first frame to its last, at 30 fps and 960 by 540
(headless, so it works with `TIDE_HIDDEN`); it needs Tide 0.5.0-nightly.45 or
later for `tide run --record`, and says so otherwise. With `--frames N` the GIF
loops the clip for `N` frames at 60 fps instead. `TIDE_HIDDEN` previews require
`--frames`, `--capture` or `--gif`.

Existing output files are refused. Every generated GLB has a JSON recipe beside
it recording prompt/input paths and generation settings: for `generate`, each
prompt with its duration and frames, the transition, placement, seed, guidance,
constraint guidance, correction frames, in-place, the device it ran on, threads,
the variant set, every constraint with its source file and frame, the model's
error there (`modelError`) and the error after correction (`error`). Paths and prompts remain local: this tool sends
no telemetry and starts no server.

## Batch actions

```sh
tide-motion batch examples/jobs.json
```

The [example batch](examples/jobs.json) generates two actions and connects them.
Paths in a manifest are relative to that manifest. Actions run in order, so later
actions can consume earlier outputs. Kimodo and MotionBricks each load lazily on
their first use and remain loaded until the complete batch finishes; model
checksums are verified only once per backend. All native resources are released
when the command ends.

Batch mode also keeps Kimodo's text-encoder layers resident by default, avoiding
repeated weight reads between prompts. This needs more RAM than streaming the
layers. An explicit `KIMODO_TEXT_LAYER_CHUNK` environment setting is honored for
machines that need the upstream streaming behavior. Work remains sequential;
batching does not mean concurrent inference or tensor batching.

The tool validates the manifest and output paths before loading a model. It stops
at the first failed action and retains completed outputs. A rerun requires new
output paths or a manifest containing only the unfinished actions.

Retargeting can appear anywhere in the same ordered batch, using a previous
generated or connected output. It does not load an inference backend:

```json
{
  "action": "retarget",
  "source": "sequence.glb",
  "target": "character.glb",
  "map": "humanoid.json",
  "output": "character-sequence.glb"
}
```

For the reverse direction, use `"toSoma": true` and omit `target`. `clip` selects
a source clip. Mapping schemas and required fields are checked before any batch
action runs. As with other actions, all file paths are relative to the manifest.

Generate actions take `prompt` and `duration`, or `prompts` for a sequence, plus
every constraint option:

```json
{
  "action": "generate",
  "prompts": [
    { "prompt": "A person swings a sword horizontally from right to left with both hands", "duration": 1.5 },
    { "prompt": "A person steps back into a guard stance", "duration": 1.5 }
  ],
  "transition": 8,
  "startPose": "guard.glb",
  "endPose": { "file": "guard.glb", "frame": 0 },
  "keyframes": [ { "frame": 45, "file": "side-cut.glb", "sourceFrame": 30 } ],
  "effectors": [ { "effectors": ["right-hand"], "frame": 30, "file": "side-cut.glb", "sourceFrame": 30 } ],
  "path": [ { "frame": -1, "x": 0, "z": 0.5, "yaw": 0 } ],
  "placement": "auto",
  "correct": true,
  "seed": 5,
  "output": "sequence-from-guard.glb"
}
```

Every generate option has a field: `variants`, `variantsGif`, `guidance`,
`constraintGuidance`, `correctionFrames`, `inPlace`, `device` (`"auto"`,
`"gpu"`, `"cpu"`), `threads` and `tide`; `connect` takes `threads` too. Variant
outputs and their sheet are checked for collisions with every other output
before anything runs. A `mirror` action takes `source`, optional `clip` and
`output`:

```json
{ "action": "generate", "prompt": "A person kicks forward with the right leg", "duration": 2.5,
  "variants": 8, "guidance": 3, "startPose": "idle.glb", "endPose": "idle.glb", "output": "kick.glb" },
{ "action": "mirror", "source": "kick.seed5.glb", "output": "kick-left.glb" }
```

A pose is a file path (default frame as on the command line) or `{ "file", "frame" }`;
`effectors` is a list or a comma-separated string. Pose files may be outputs of
earlier actions, so only their paths are checked before the batch starts; their
frames are checked when the action runs.

## Build

Requires CMake 3.25+, Git and a C++23 compiler. Dependencies are fetched at pinned
revisions; `patches/kimodo-conditioning.patch` is applied to Kimodo when it is
fetched, exposing its conditioned sampler. Model conversion is not part of the build.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL
cmake --build build --config Release --target tide-motion motion-tests retarget-tests
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release --component runtime --prefix dist/tide-motion
```

On Unix use Ninja and `-DCMAKE_BUILD_TYPE=Release`. CPU inference is the portable
default. `-DMOTION_VULKAN=ON` enables Kimodo's Vulkan backend and requires a Vulkan
SDK **at build time** (the Windows release uses it). The built CLI falls back to the
CPU without a usable Vulkan device; on Windows the loader is delay-loaded, so it
also starts without one. This option does not accelerate MotionBricks, which uses
ONNX Runtime CPU inference.

See [implementation notes](notes/implementation.md) for exact revisions, current
validation, remaining gaps and platform coverage.

`tests/install.ps1 -Archive <release.zip>` and
`sh tests/install.sh <release.tar.gz>` check installers in isolated directories.
`tests/inference.ps1 -Models <cache> [-Tide <executable>]`
runs a real five-action generation/transition/constrained-sequence batch, checks
the constraint errors in the recipe and model reuse, and with `-Tide` records a
preview PNG and GIF.
The latter needs the downloaded models and can take several minutes on CPU.

## Releases

The [Build and release workflow](.github/workflows/release.yml) builds all four
platforms on pushes to `dev` and on pull requests. Each runs the model-free
contract tests, builds a runtime-only archive and checks the installer with the
packaged executable. Models are never included in CI artifacts or releases.

To publish, update the version in `CMakeLists.txt`, commit it to `dev`, then push
the matching `v<version>` tag. A tag/version mismatch fails. Only after every
platform succeeds does the workflow publish all archives, SHA-256 sidecars and
both installers together. A failed build publishes nothing; a publish retry can
resume a draft but cannot overwrite an already published release. The Windows
release is built with Vulkan; the others use CPU inference.

## Licenses

Source: Apache-2.0. Third-party code and binary notices are in `licenses/`.
Kimodo SOMA and MotionBricks SOMA model licenses permit commercial use, subject
to their respective terms. The models and text encoder retain their own licenses;
they are not relicensed as Apache-2.0 by this tool. **Built with Meta Llama 3.**

No evaluation motion samples, proprietary training data or research-only SMPL-X
checkpoints are included or downloaded. Supply source motion/characters you have
rights to use. NVIDIA's output terms do not establish copyright ownership or
clear third-party rights in user-supplied assets.

### Offline pose sequences (MotionBricks)

`tide-motion pose-sequence poses.json -o authored.glb --threads 8` generates
through explicit ordered pose targets with one native ONNX session. Start from
[examples/pose-sequence.json](examples/pose-sequence.json), replacing the clip
paths and placements with your own. Files are relative to the JSON file; use
SOMA GLBs, or first convert a manually mapped humanoid with `retarget --to-soma`.
No upstream sample or evaluation motions are supplied.

The schema is `tide-motion-pose-sequence-1`, with `start` and 1..128 `targets`.
Each entry requires `file`, `position` (absolute **hips** XYZ in Y-up metres,
including height), and `yaw` (absolute facing in degrees: 0 is +Z, 90 is +X).
`clip` selects an animation; `frame` selects a 30 Hz sample (negative counts from
the end; start defaults to -1, targets to 0). Root displacement and orientation
are rotated about that sample; local body rotations and lengths stay intact.
A start uses the four samples ending there, a target the four samples starting
there; sampling clamps at clip ends. `hold:true` instead repeats the chosen pose
four times to request zero guide velocity. It does not insert a timed hold.

Each target requires integer `tokens` 6..16. With Tide's existing three-sample
boundary trimming it adds **4*tokens-7 intervals** at 30 Hz: 8 tokens arrives 25
frames (0.833 s) later, 10 tokens 33 frames (1.1 s) later. Arrival frames accumulate
from frame 0; no resampling to arbitrary seconds is performed. All targets are
reached exactly after the existing endpoint correction. One copy of each seam is
exported. Later generations use the last four generated samples as context.
The output contains only the selected start and generated motion, without source
clip lead-ins or target clip tails.

The adjacent `.glb.json` records `authoring`: a reusable schema object with
absolute paths, resolved frames, effective positions/yaws/hold settings and token
counts, plus computed startFrame/endFrame/seconds on targets. Extract `authoring`
to a JSON file to edit and rerun; computed timing fields are recomputed on load.
Model revision, tool version, thread setting and final frame count are recorded
beside it. Sources must remain available; recipes do not embed motion data.

Batch action: `{"action":"pose-sequence","recipe":"poses.json","threads":8,
"output":"authored.glb"}`. The entire recipe, every source and all output
collisions are checked before inference, including when a later batch action is
invalid. Pose-sequence sources must already exist at batch preflight; an earlier
batch action cannot create them in the same run. Interleaved connect actions
reuse the session when thread settings agree. Existing connect/batch schemas stay
compatible. Limits: 1 MiB JSON, 128 targets, 18,000 exported frames and existing
GLB limits. Failure keeps completed batch assets.

Design references are the pinned upstream [Inbetweener](https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/docs/inbetweener.md)
and [Smart Primitive Editor](https://github.com/NVlabs/motionbricks/blob/54fc563aeaaf819a753d72cc9fdc6713989cdd43/docs/smart_primitive_editor.md):
ordered pose windows and explicitly placed interaction targets. This feature has
no objects, collision detection, scene/contact understanding, automatic placement,
pivots, early transitions, IK or foot cleanup. Pose continuity is exact; velocity,
acceleration, foot planting and plausible motion between distant/contradictory
poses are not guaranteed. Inspect the generated GLB in a bounded Tide preview.

## Optional foot contact cleanup

`cleanup` applies native two-bone foot IK to an existing generated or retargeted
clip. Generation and retarget defaults stay unchanged; no model is loaded.

```powershell
tide-motion cleanup idle.glb --contacts contacts.json -o idle-clean.glb
# A custom rig needs its explicit SOMA mapping:
tide-motion cleanup character-idle.glb --map examples/synty-polygon.json --contacts contacts.json -o character-clean.glb
```

Copy `examples/contacts.json` and edit the **inclusive, zero-based** frame intervals
for actual planted feet. Each interval anchors the ankle to its first sample in
world space. There is no automatic contact or ground detection. Do not label a
kick, airborne motion, a deliberate sliding step or a foot pivot as a contact.
Only SOMA-30 and mapped direct thigh/calf/ankle chains are supported; intermediate
helpers in those chains are refused. Mapping does not identify bones automatically.

`fadeFrames` is 0..60 (default 3), a raised cosine fade at either interval edge
and near protected poses; 0 applies the lock immediately. `maxCorrection` is the
maximum distance in metres from the first ankle sample, in (0,0.5] (default 0.1).
An interval exceeding it is skipped entirely. A solved interval whose mean
horizontal drift increases by more than 0.1 micrometres per frame is also rolled
back in full (`skipReason`: `correction-bound` or `drift-increased`). Same-foot intervals cannot overlap;
both feet may be planted simultaneously. Each interval needs at least three frames.
This limit bounds ankle displacement, not the joint angle. Review the reported
joint corrections, especially on nearly straight knees.

Root motion and all local translations stay unchanged. Foot world orientation is
kept; toes can still articulate. No ground projection, pelvis adjustment, joint
limits, collision solver or sole/toe locking is performed. An unreachable ankle
has residual error rather than stretching its bones or moving the root. Fades,
protected poses and unreachable targets can leave drift or increase movement near
an interval boundary. These are explicit editing controls, not a quality guarantee.

Every constraint frame in an adjacent source `.glb.json` recipe is protected
exactly and retained for subsequent cleanup. For externally authored clips or
constraints lost through other conversions (including retarget), put their frame
indices in `protectedFrames`. All joints at those frames remain unchanged.
Generated effector/path targets are still measured by generation; this command
does not solve those targets or change their existing defaults.

The output recipe records `tide-two-bone-v1`, the effective settings, explicit
intervals, protected frames, mapping (if used), source recipe, skipped intervals,
mean horizontal ankle movement per frame before/after (metres), maximum distance
from the anchor after cleanup, and maximum local joint rotation correction
(degrees). Metrics include fades and protected frames. A saved asset can contain
skipped intervals; inspect its recipe.

Batch action (paths relative to the manifest; the plan and mapping are preflighted
before any model loads):

```json
{"actions":[{"action":"cleanup","source":"idle.glb","contacts":"contacts.json","output":"idle-clean.glb"}]}
```

Add `map` and/or `clip` for a mapped rig or named animation. Place this action after
`generate`, `connect` or `retarget` to postprocess that action's output. Each output
remains an ordinary GLB skeleton and animation, without mesh or skin changes.
