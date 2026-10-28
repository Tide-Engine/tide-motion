# ARDY native offline generation

ARDY is an opt-in native CPU/Windows DirectML backend. It uses the commercially released SOMA
KimodoRep checkpoint at `e6c6804fb6b55636ad927145d74959fd46882ccd`, converted from
[NVIDIA's pinned source](https://github.com/nv-tlabs/ardy/tree/e706e27acc88070918308a10922b8f7174d0825e).
It generates motion with the complete two-stage denoiser, FSQ tokenizer
encoder/decoder, local LLM2Vec embeddings and a native DDIM/autoregressive host.
No Python, text server, simulation or procedural motion generator runs in the
installed executable. The initial [feasibility assessment](ardy-feasibility.md)
describes the architecture; its earlier scope/budget/speed gates are superseded
by the owner's implementation authorization. No speed multiplier is a requirement.

## Use

```powershell
tide-motion models install text --accept-licenses
tide-motion models import-ardy PATH/TO/CONVERTED --accept-licenses
tide-motion models check ardy
tide-motion generate 'A person walks forward at a relaxed pace.' --backend ardy `
  --device gpu --duration 6.1 --steps 10 --threads 8 --no-correction -o walk.glb
tide-motion generate 'A person walks forward.' --backend ardy --duration 2 `
  --then 'A person waves with their right hand.' --duration 2.1 `
  --history walk.glb --history-frames 64 --no-correction -o continuation.glb
```

`--models DIR` selects the common model root. Text GGUF files are shared on disk
with Kimodo; converted files live in `DIR/ardy/`. `models install all` retains its
existing default downloads; it does not fetch unpublished converted ARDY weights.
The supported model-import path verifies the pinned manifest, model sizes/hashes
and bundled license before copying. No converted weights are hosted or included
in Git/releases by this work. Conversion is a developer step described below.

Generation defaults to 10 diffusion steps for ARDY (range 1..10). Windows x64
builds include DirectML for all three motion networks: `--device gpu` requires
successful GPU initialization/inference, `cpu` selects ORT CPU, and the default
`auto` tries GPU then falls back to CPU. A native GPU run failure in `auto`
restarts the whole generation on CPU with the same seed; partially generated
motion is discarded. Both paths use native CPU text embeddings. Other platforms
currently have only the CPU path; an explicit GPU request is refused there.
The native dependency's Vulkan support for Kimodo is unaffected. ARDY defaults
to 8 CPU threads, and batch actions reuse all three sessions and cached text
embeddings while their thread/device setting stays the same. Recipes record the
actual device and `accelerator:directml|cpu`, including after a fallback.

Per-prompt duration may be 0.2..600 seconds; the total remains bounded to 18,000
frames (10 minutes at 30 Hz). The requested output count is rounded seconds×30,
including partial last chunks. Prompt changes are selected at the start of each
60-frame generation chunk, so a change inside a chunk takes effect at the next
chunk. No interpolation or pose blend bridges ARDY prompts. `--transition` is
refused for this backend. The existing start/end/keyframe/effector/path options
are encoded in the checkpoint's own feature space and masks. Constraint errors
are reported before optional existing pose correction. Use `--no-correction` to
inspect the actual generated output; contacts and constraints are not guarantees.

`--history` takes a canonical SOMA GLB; the last up to `--history-frames` samples
are encoded by the learned tokenizer. Visible history is 4..240 frames, a multiple
of four (default 240). A partial history token is discarded from the beginning;
at least four samples are required. The exported clip contains newly generated
samples only. Import does not promise an exact first-frame pose or velocity seam.
For explicit first-frame pose requirements, use the existing pose constraints.

The same fields work in batch actions: `backend:"ardy"`, `history`,
`historyFrames`, `prompts`, `duration`, `steps`, `seed`, `threads` and existing
constraint/variant controls. Paths are manifest-relative. Backend-specific invalid
steps, devices and history settings are rejected during batch preflight.
Recipes record backend, pinned checkpoint and manifest hashes, history settings,
chunk length, sampler, effective CPU threads, prompts, seeds and constraint errors.
Imported history is referenced by path, so preserve the source asset to reproduce it.

## Model and native contracts

The converted bundle is three self-contained opset-17 ONNX graphs, not an export
presented as a backend. The executable evaluates those graphs through the existing
pinned ORT 1.31.1 C++ runtime. C++ owns seeded Gaussian noise, cosine schedule and
timestep mapping, DDIM updates, chunk selection, history/future masks, origin
translation, FSQ history requantization, context cropping, prompt caching and
canonical GLB export. Graphs include learned layers and their root/latent feature
conversions. All graph shapes are fixed at 75 tokens/300 frames; padding masks
permit 0..240 history frames and a 60-frame generation chunk. Remaining slots
carry future constraints. Future constraints outside that window are reached as
the window advances; they are not silently allocated as unlimited context.

SOMA hierarchy, names and neutral offsets were compared against the pinned
upstream skeleton. Output uses 30 Hz canonical SOMA parent-local rotations/rest
translations and the real pelvis position (smooth root XZ plus predicted pelvis
offset, absolute pelvis Y). It reuses the existing finite-transform/FK/GLB path
and explicit mapped humanoid retargeting. No automatic bone identification,
skinning, non-humanoid retargeting or match-simulation determinism is implied.

Imported history uses the upstream ADMM smooth-root preprocessing, including its
final half-sample interpolation, 30 Hz FK velocities and contact classification.
Tiny arithmetic differences can cross a finite scalar quantization boundary.
The separately reported preprocessing and fixed-input host comparisons distinguish
that sensitivity from a host-loop implementation error. Native RNG sequences are
not assumed identical to PyTorch or portable bit-for-bit across standard libraries.
Reference host tests use captured fixed Gaussian arrays instead of seed equality.

## Licenses

The [pinned checkpoint LICENSE](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/blob/e6c6804fb6b55636ad927145d74959fd46882ccd/LICENSE)
is the October 24, 2025 NVIDIA Open Model License Agreement. Its section 2.2
expressly grants use, derivative-work creation and distribution, including
commercial rights; section 3.1 requires the agreement and Notice attribution for
model distribution. The model card links a newer commercial agreement. We rely
on the actual pinned bundled grant, retain its conditions (including guardrail,
Trustworthy AI and trade terms) and include it plus the required Notice in the
conversion/import. No missing permission for this conversion/native inference
was identified in that bundled grant. The differing version is recorded and is
not treated as a technical blocker or represented as legal clearance for all uses.
No upstream guardrail is removed or bypassed. No outside party was contacted.

`licenses/NVIDIA-ARDY-SOURCE.txt` preserves upstream Apache-2.0/McGill MIT terms;
`licenses/NVIDIA-ARDY-MODEL.txt` preserves the actual model grant. The local text
encoder retains existing Meta Llama 3/model-adapter terms and attribution. No
research-only checkpoint or evaluation motion data is used. Generated GLBs remain
ordinary animation assets; NVIDIA's model terms claim no ownership of outputs.

## Native Windows GPU runtime

The packaged backend uses the native [ORT DirectML execution provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
and DirectX 12, with no user CUDA, Python, .NET, SDK or service dependency.
Windows 10 version 1903 or newer and a suitable DirectX 12 GPU/driver are required
by DirectML. The backend chooses the high-performance hardware adapter; there
is no adapter-index option. Only Windows x64 / NVIDIA RTX 4080 SUPER has been
validated here; AMD/Intel hardware and other platforms are untested.

Pinned native dependencies, downloaded and SHA-256-checked by CMake:

| Package | Version | Archive SHA-256 | Terms |
|---|---|---|---|
| [Microsoft.ML.OnnxRuntime.DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML/1.24.4) | 1.24.4, source `2d924974ef147392ced8409d36bd6d2e7fcc8a74` | `57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681` | MIT + ThirdPartyNotices |
| [Microsoft.AI.DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) | 1.15.4 | `4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9` | Microsoft DirectML terms + ThirdPartyNotices |

DirectML's bundled section 1(a) permits redistribution within developed Windows
applications, with Windows/Xbox-only use. The package retains both original
licenses and third-party notices under `licenses/ardy-ort-directml` and
`licenses/directml`. The Microsoft SDK is proprietary; ORT is MIT. This change
does not alter the pinned NVIDIA checkpoint grant or distribute any weights.
ORT lists DirectML as sustained engineering, a maintenance/platform risk for a
future dependency update; this is a working optional path, not a portable GPU API.

CPU ORT remains 1.31.1 for ARDY and MotionBricks. The GPU distribution has its own
ORT 1.24.4 API table in `src/ardy_gpu.cpp`, loaded with an explicit absolute path
from `bin/ardy-directml`. No ORT C++ global API, allocator, tensor or session crosses
between versions; only owned numeric vectors do. This avoids downgrading other
backends. The GPU path uses sequential execution and disabled memory patterns,
as required by DirectML, and opts out of ORT telemetry through its native API.
The three DLLs add 35,877,968 uncompressed bytes. There is no runtime download.
`-DMOTION_ARDY_DIRECTML=OFF` builds/packages the existing CPU-only path.

All three unchanged FP32 model graphs run on GPU, including the FSQ encoder and
decoder. DDIM, masks, history, origins, preprocessing, FK and GLB export remain
native C++; local GGML text stays on CPU. Developer `TIDE_ARDY_PROFILE=PATH_PREFIX`
enables ORT JSON profiling. Actual profiles on the tested GPU contain DML kernel
events for every network and no CPU execution-provider kernel events. Model
loading or ONNX export alone is not the evidence for acceleration.

## Reproduce conversion and reference checks

Developer tooling is excluded from the runtime package. Use an isolated Python
3.11 environment; `tools/ardy/requirements-dev.txt` pins the environment exercised
here (including CPU PyTorch 2.7.1, transformers 5.8.1, FSQ 1.25.2 and ONNX 1.18.0).
Install CPU PyTorch from its official CPU wheel index before the requirements.
The supplied prepare script pins and hashes configuration, both safetensors files,
all statistics and the actual license; the export also verifies them before use.

```powershell
$env:TIDE_HIDDEN='1'
uv venv .research/ardy/venv --python 3.11
uv pip install --python .research/ardy/venv/Scripts/python.exe torch==2.7.1 `
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .research/ardy/venv/Scripts/python.exe -r tools/ardy/requirements-dev.txt
git clone https://github.com/nv-tlabs/ardy.git .research/ardy/upstream
git -C .research/ardy/upstream checkout e706e27acc88070918308a10922b8f7174d0825e
python tools/ardy/prepare.py .research/ardy/checkpoint --accept-licenses
./.research/ardy/venv/Scripts/python.exe tools/ardy/export.py `
  --source .research/ardy/upstream --checkpoint .research/ardy/checkpoint `
  --output .research/ardy/native
```

The pinned bundle manifest SHA-256 is
`aac3b966ed1368c4ec126c969f0763d59126b047a1053a2394c20983acb917d2`.
Its graph hashes are checked by native code; a different conversion must be
reviewed/repinned rather than silently accepted. Python export-time ORT 1.23.2
checks are followed by tests against the actual native ORT 1.31.1 binary.
Export trace warnings about fixed shapes are expected; no dynamic-shape claim
is made. Export fails on reference mismatches rather than logging success anyway.

Build normally, then run native `ardy-tests MODEL_DIR REFERENCE_DIR [cpu|gpu]` against
generated fixtures. `ardy-tests MODELS/ardy --text TEXT.json` emits the native
local embedding for an own prompt. `tools/ardy/host_reference.py` uses that exact
embedding with the upstream PyTorch sampler and captures fixed noise; its optional
`--history-glb` uses your generated GLB, not evaluation data. This validates
motion networks/host integration independently of the text model's Q8-versus-BF16
quantization difference. It does not claim full FP32 LLM2Vec text-encoder parity.

`tests/ardy-inference.ps1` exercises cold/warm batch actions, changing prompts,
imported history, sparse constraints and a 60-second output. It holds
`TideMotionGpuValidation`, records wall time and peak process working set, and
checks requested counts through the native GLB loader. `ardy-tests --assets DIR`
checks FK finiteness, same-seed warm repeatability and GLB roundtrip and records
travel/foot-slide diagnostics. Never package generated fixtures or weights.
Pass `-Device gpu` or `-Device cpu` to the inference harness for matched runs.
`tests/ardy-runtime.ps1 -Tool PACKAGED_EXE -Models MODEL_ROOT -OutputDirectory NEW_DIR`
tests real GPU inference with only Windows directories on PATH, mixed GPU ARDY /
CPU MotionBricks sessions when that model is supplied, and actual CPU fallback
versus strict-GPU refusal in a private deployment missing the DirectML DLLs.
Both scripts hold `TideMotionGpuValidation`. Use a fresh output directory for
the runtime script. Model-free CTests never require a GPU or download weights.

## Validation and limits

Windows x64 Release, Visual Studio ClangCL 18.1.8; ARDY C++ builds with warnings
as errors. All six model-free CTests pass. Native fixed-input tests exercised
0/4/240 history frames and constraints, with maximum denoiser absolute error
2.08e-5 and decoder 1.54e-3 (mean 1.09e-5) against PyTorch. FSQ encoder fixtures
match exactly. A four-chunk, 183-frame fixed-noise path-constrained host test
matches decoded root positions within 0.056 mm. Imported-history fixed-input
encoder/host comparison is within 0.26 mm; including native FK/smoothing gives
2.2 mm and 1.84 degrees in this test. Those are observed reference differences,
not a universal quality bound or cross-platform determinism claim.

A native own-prompt 183-frame walk was exported and rendered on the mannequin
in a bounded 91-frame capture. Tide CLI and `tide_platform` were rebuilt from
`D:/Projects/PurrEngine` at `28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c` and staged
privately; TIDE_HIDDEN=1, with the GPU mutex held for capture. No engine source
changed. The inspected front/profile frame shows a walking pose with intact
limbs and floor alignment; a frame capture is not proof of continuous contact
quality. This original CPU validation does not establish Linux/macOS support;
subsequent Windows GPU evidence is below. The networks can produce skating, semantic misses and
constraint error. Text model loading/encoding remains significant for cold runs;
interleaved Kimodo/ARDY actions currently own separate text encoder instances.

Measured cold/warm/long-form and runtime-package evidence is recorded in
notes/implementation.md after local validation. ARDY supplements prompt-changing
long-form continuation; current Kimodo generation and MotionBricks clip/pose
authoring remain available. No universal quality or speed superiority is claimed.

### Windows GPU validation (2026-10-28)

Windows x64 ClangCL 18.1.8 Release, RTX 4080 SUPER (16 GB, driver 617.42), Ryzen
9 7900X, eight CPU threads, same external models, prompts, seeds and no pose
correction. All ten integrated CTests pass with DirectML enabled and in a separate CPU-only
build. Native GPU reference tests cover every network at 0/4/240 history frames,
full host generation across four chunks with a future path constraint, and
imported-history preprocessing/FSQ. Maximum network errors against PyTorch:
denoiser 0.000138, encoder 0.000000477, decoder 0.001622. The original tensor
tolerance (0.002 absolute + 0.002 relative) is unchanged. Fixed-input host errors
are 3.55 mm / 0.783 degrees; imported-history host 0.258 mm / 0.194 degrees;
including native preprocessing 2.18 mm / 1.85 degrees. GPU decoded host tests use
5 mm / 0.025 radians instead of CPU's 3 mm / 0.01 radians because different GPU
reductions can cross discrete FSQ boundaries. This is measured parity, not bitwise
CPU/GPU equivalence or a universal error bound.

Matched single-process five-action batches, one run per device:

| Work | CPU seconds | DirectML seconds |
|---|---:|---:|
| Cold own-prompt 183 frames, 10 steps | 18.010 | 14.673 |
| Warm same prompt/seed, 183 frames, 10 steps | 5.238 | 0.221 |
| Imported 64-frame history, changing prompts, path, 123 new frames | 4.503 | 0.576 |
| Full pose, right hand and future path constraints, 183 frames | 4.925 | 0.221 |
| Warm 60-second / 1,800-frame generation, 4 steps | 14.835 | 0.700 |
| Whole process including session loads, text and GLB writes | 49.480 | 19.670 |

Per-generation times include prompt encoding and exclude model session creation;
warm means cached sessions/embeddings in that batch. Cold is process-cold, not an
OS disk-cache flush. This is one machine/run, not a portable benchmark. Peak
sampled process working set was 9.18 GB CPU / 9.48 GB GPU, sampled every 250 ms.
DXGI per-process local GPU usage sampled after runs peaked at 983,146,496 bytes;
this excludes other processes and can miss transient peaks. Text loading/encoding
still dominates cold latency and host RAM. No text GPU acceleration is claimed.

Both batches pass exact requested frame counts, finite FK, native GLB roundtrip
and same-device cold/warm pose repeatability. Constraints on GPU measured 1.57 cm
mean / 4.01 cm max full-pose position error, 1.45 cm right-hand error and 3.28 cm
path error before correction. GPU and CPU are not interchangeable recipes: their
same-seed four-step 60-second runs traveled 18.87 m and 22.57 m respectively as
quantization differences accumulated. Foot-slide diagnostics found no samples
below the strict 3 cm height threshold (null, not zero sliding). Walking, waving
and the long-run poses were inspected in bounded 91/181/1801-frame hidden Tide
captures using the CLI/platform built from engine checkout
`28b61e4db1cb2549fcc9f6bc6d9c6034022f3d9c`. Static captures do not prove contact or
all-frame semantic quality.

The fresh ZIP installs/reinstalls in a path with spaces and quotes. Packaged
native GPU generation, CPU MotionBricks in the same process, explicit-GPU failure
without DLLs, and actual auto CPU generation all pass with a Windows-only PATH.
All inference/timing/captures were serialized by `TideMotionGpuValidation`.
Runtime loss mid-generation has a CPU-restart handler but physical device-reset
fault injection was not performed. No AMD/Intel GPU, Linux/macOS GPU, multi-GPU
selection, text offload or broad driver compatibility claim is made.
