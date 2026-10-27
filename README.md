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
./install.ps1 -Archive ./tide-motion-0.2.0-windows-x64.zip -AddToPath
```

Reopen the terminal after adding PATH. `-Root <directory>` selects another
installation. Linux/macOS (select your platform's archive):

```sh
sh install.sh ./tide-motion-0.2.0-linux-x64.tar.gz
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
minutes on its CPU. `KIMODO_BACKEND=cpu` forces the CPU. MotionBricks (`connect`)
always runs on the CPU. Other platforms' packages are CPU-only for now.

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
tide-motion preview sequence.glb
tide-motion preview sequence.glb --tide /path/to/tide --capture preview.png --frames 90
```

Preview makes a temporary Tide project and draws the animated skeleton from front
and side. It uses a separately installed Tide; no engine is bundled. Captures use
Tide's ordinary bounded capture mode. `TIDE_HIDDEN` previews require `--frames`
or `--capture`. General game-character previews remain future work.

Existing output files are refused. Every generated GLB has a JSON recipe beside
it recording prompt/input paths and generation settings. Paths and prompts remain
local: this tool sends no telemetry and starts no server.

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

## Build

Requires CMake 3.25+, Git and a C++23 compiler. Dependencies are fetched at pinned
revisions. Model conversion is not part of the build.

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
runs a real four-action generation/transition batch and verifies model reuse.
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
