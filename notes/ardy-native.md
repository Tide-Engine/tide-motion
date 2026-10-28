# ARDY native offline generation

ARDY is an opt-in native CPU backend. It uses the commercially released SOMA
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
  --duration 6.1 --steps 10 --threads 8 --no-correction -o walk.glb
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

Generation defaults to 10 diffusion steps for ARDY (range 1..10); `--device auto`
and `cpu` both use ORT CPU and native CPU text. `gpu` is explicitly refused.
The native dependency's Vulkan support for Kimodo is unaffected. ARDY defaults
to 8 CPU threads, and batch actions reuse all three sessions and cached text
embeddings while their thread setting stays the same.

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

Build normally, then run native `ardy-tests MODEL_DIR REFERENCE_DIR` against
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
quality. No ARDY GPU acceleration or Linux/macOS support is claimed from this
Windows CPU validation. The networks can produce skating, semantic misses and
constraint error. Text model loading/encoding remains significant for cold runs;
interleaved Kimodo/ARDY actions currently own separate text encoder instances.

Measured cold/warm/long-form and runtime-package evidence is recorded in
notes/implementation.md after local validation. ARDY supplements prompt-changing
long-form continuation; current Kimodo generation and MotionBricks clip/pose
authoring remain available. No universal quality or speed superiority is claimed.
