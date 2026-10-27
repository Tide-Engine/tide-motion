# Tide Motion

Native, offline animation authoring using Kimodo for text-to-motion and
MotionBricks for transitions between clips. It is a separate tool and repository;
Tide is an optional preview renderer. No Tide CLI or engine changes are required.

This initial implementation targets canonical SOMA-30 human animations. Outputs
are self-contained GLB files containing an animated skeleton, without a mesh.
Other rigs need retargeting in an authoring tool before/after this workflow.
Automatic rigging, skinning and general retargeting are not implemented.

## Install

Windows x64 (AVX2 CPU): download the release ZIP and its `.sha256` file, then run:

```powershell
./install.ps1 -Archive ./tide-motion-0.1.1-Windows-AMD64.zip -AddToPath
```

Reopen the terminal after adding PATH. `-Root <directory>` selects another
installation. The archive includes the native executable, ONNX Runtime and
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

Inputs must use the canonical SOMA names, hierarchy and proportions, with no
scale tracks, matrix nodes or animated extra nodes. LINEAR and STEP GLB tracks
are supported; bake cubic tracks before import. A GLB with several animations
requires `--from-clip NAME` or `--to-clip NAME` (`--clip NAME` for inspect/preview).

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

## Build

Requires CMake 3.25+, Git and a C++23 compiler. Dependencies are fetched at pinned
revisions. Model conversion is not part of the build.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL
cmake --build build --config Release --target tide-motion motion-tests
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release --component runtime --prefix dist/tide-motion
```

On Unix use Ninja and `-DCMAKE_BUILD_TYPE=Release`. CPU inference is the portable
default. `-DMOTION_VULKAN=ON` enables Kimodo's optional Vulkan backend and requires
a Vulkan SDK **at build time**. This option does not accelerate MotionBricks,
which uses ONNX Runtime CPU inference. GPU-enabled builds need separate validation.

See [implementation notes](notes/implementation.md) for exact revisions, current
validation, remaining gaps and platform coverage.

`tests/install.ps1 -Archive <release.zip>` checks the Windows installer in an
isolated directory. `tests/inference.ps1 -Models <cache> [-Tide <executable>]`
runs a real four-action generation/transition batch and verifies model reuse.
The latter needs the downloaded models and can take several minutes on CPU.

## Licenses

Source: Apache-2.0. Third-party code and binary notices are in `licenses/`.
Kimodo SOMA and MotionBricks SOMA model licenses permit commercial use, subject
to their respective terms. The models and text encoder retain their own licenses;
they are not relicensed as Apache-2.0 by this tool. **Built with Meta Llama 3.**

No evaluation motion samples, proprietary training data or research-only SMPL-X
checkpoints are included or downloaded. Supply source motion/characters you have
rights to use. NVIDIA's output terms do not establish copyright ownership or
clear third-party rights in user-supplied assets.
