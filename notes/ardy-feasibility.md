# ARDY native feasibility spike — 2026-10-28

Historical spike, superseded by [the implemented native backend](ardy-native.md).
The original budget and speed/value gates were not owner requirements and are
withdrawn. Commercial SOMA weights exist and their architecture
looks implementable with native tensor operations. This spike validates container
loading and one real learned projection, not generation, complete operator coverage,
quality, speed, or portability. No CLI backend, model installer entry, runtime
dependency or engine change is added. Baseline reviewed: v0.5.0 / dd4dc0b.

## Sources and exact candidates

All source inspection is pinned to [nv-tlabs/ardy at
e706e27acc88070918308a10922b8f7174d0825e](https://github.com/nv-tlabs/ardy/tree/e706e27acc88070918308a10922b8f7174d0825e).
The [project](https://research.nvidia.com/labs/sil/projects/ardy/) and
[paper](https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf)
explain its hybrid root/body representation and streaming objective. Paper/demo
performance is not a measurement of these SOMA checkpoints in Tide.

Only these two SOMA RP candidates are in scope (both 30 joints, 30 Hz, horizon 60):

| Candidate | Hugging Face revision inspected | Representation |
| --- | --- | --- |
| [ARDY-SOMA-RP-30FPS-Horizon60](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-Horizon60/tree/244215084b22fcd29ab743017330e87411fe168a) | `244215084b22fcd29ab743017330e87411fe168a` | Raw hip root, 366 explicit features |
| [ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/tree/e6c6804fb6b55636ad927145d74959fd46882ccd) | `e6c6804fb6b55636ad927145d74959fd46882ccd` | Smooth root, 369 explicit features |

Prefer KimodoRep for the first prototype because it reduces representation work.
Do not mix its statistics, tokenizer or weights with the raw-hip candidate, Kimodo,
or MotionBricks. Similar features do not mean interchangeable learned models.
Core/G1 releases are different rigs/FPS and unnecessary here. No research-only
checkpoint or evaluation dataset was downloaded or used.

## License finding — preserve the actual files

Both model cards explicitly say commercial and noncommercial use is allowed. But
the cards link the [April 2, 2026 NVIDIA Open Model
Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
whereas the exact [raw-hip LICENSE](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-Horizon60/blob/244215084b22fcd29ab743017330e87411fe168a/LICENSE)
and [KimodoRep LICENSE](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/blob/e6c6804fb6b55636ad927145d74959fd46882ccd/LICENSE)
contain **NVIDIA Open Model License Agreement, Version Release Date October 24,
2025**, referring to `/nvidia-open-model-license/`. This is a real discrepancy,
not a research-only restriction. Both texts permit commercial use, but they are
not the same terms. Use the bundled terms conservatively and resolve the
card/file discrepancy with upstream before adding a model distribution manifest.
Do not silently substitute Tide's existing NVIDIA model notice.

The bundled 2025 text has guardrail/updated-terms conditions, Trustworthy AI
terms, an indemnity clause and trade compliance requirements. Section 3.1 requires
a copy of that agreement and a Notice attribution when distributing the model.
It states outputs are not derivative models and claims no ownership of outputs.
Cosmos-specific attribution is conditional; these cards identify ARDY, not Cosmos.
This is an engineering inventory of the supplied terms, not a blanket clearance
of third-party components. Keep future converted weights external, pinned by
revision, size and SHA-256, behind explicit model-license acceptance.

Upstream [source LICENSE](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/LICENSE)
is Apache-2.0 with McGill LLM2Vec MIT notices; individual files carry notices.
[ATTRIBUTIONS.MD](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ATTRIBUTIONS.MD)
also identifies Unitree BSD material. Do not vendor unrelated skins/robot assets.
The native probe is independently written, incorporates no ARDY code and reuses
Tide's pinned nlohmann/json 3.12.0 (MIT). A future port must retain all applicable
notices and mark modifications. Llama model terms remain separate from MIT code.

## Checkpoint architecture and operators

The pinned [KimodoRep config](https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/blob/e6c6804fb6b55636ad927145d74959fd46882ccd/config.yaml)
sets a 10-step base schedule, separated text/constraint CFG, a 60-frame (2 s)
generation horizon, four frames per token and 128 latent body values. One hybrid
token therefore has 20 explicit root values plus 128 latent values = 148.
There are 15 newly generated tokens per full chunk. This is continuous diffusion
over hybrid tokens, followed by finite scalar quantization, not language-token
sampling and not a clip blend.

The [two-stage denoiser](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/auto_latent_twostage_denoiser.py)
predicts global root first, then body with a local-root condition. Each stage has
8 transformer layers, width 1024, feedforward 2048, 8 heads and GELU; dropout is
zero. Text input is one 4096-value embedding. Heading and time embeddings,
learned position prefixes relative to the first generation token, constraint
projections and frame/token masks are part of the graph.

The [tokenizer](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/autoencoder/fsq.py)
has width 512, 8 layers, 4 heads, feedforward 1024, causal encoder/decoder and
128 scalar channels with 64 levels each. Decoding unnormalizes latent statistics,
clamps to [-1,1], rounds on the half-width 32 grid and decodes conditioned on the
four local-root features per frame. It returns local root/body features; the
hybrid converter uses the explicitly predicted global root with decoded body.
The encoder is also needed to continue from an arbitrary imported pose history.

Source-derived operator inventory (not runtime coverage): affine/GEMM, residual
adds, LayerNorm, GELU, scaled dot-product attention, boolean/additive/causal masks,
reshape/transpose/concatenate/pad/slice/gather/embedding, reductions, trig/atan2,
rotation matrices and cumulative root conversions, clamp/round, normalization,
and a seeded Gaussian generator with DDIM schedule arithmetic. See
[backbone](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/backbone.py),
[autoencoder transformers](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/autoencoder/transformer.py),
[latent conversion](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/latent_utils.py),
and [diffusion](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/diffusion.py).
Mask polarity, position offsets, ties-to-even rounding and FP16 quantization
boundaries need parity fixtures, not just operator-name matching.

## Representation and GLB contract

[ArdyMotionRep](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/motion_rep/reps/ardy_motionrep.py)
has 3 raw root XYZ + 2 heading cos/sin + 29×3 non-root positions + 30×6 global
rotations + 30×3 velocities + 4 foot contacts = 366. Positions subtract hip XZ
but retain absolute height. [KimodoMotionRep](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/motion_rep/reps/kimodo_motionrep.py)
keeps all 30 positions relative to a smoothed root XZ, retaining pelvis offset
and absolute heights: 369 features, as in Tide's existing Kimodo representation.
Smooth-root Y is raw hip height. History encoding uses the upstream smooth-root
solver; simply copying raw hips is not equivalent for a moving sequence.

Decode each checkpoint using its own mean/std. Orthonormalize the global 6D
rotations, convert global to parent-local matrices/quaternions, and use canonical
rest offsets. In KimodoRep the actual pelvis is smooth root XZ plus pelvis-row
XZ offset, with pelvis-row Y; exporting the smooth root directly would lose sway.
Raw-hip ARDY uses its explicit root directly. Verify skeleton names, order,
parents, rest orientation, handedness, metres and Y-up against Tide before reuse;
30 joint counts alone do not prove rig equality. Reuse the ordinary GLB exporter
only after that check and numeric FK roundtrips. Emit 30 Hz local TRS with
quaternion sign continuity; contacts are diagnostic predictions, not planted-foot
guarantees. No 20/25 Hz resampling is needed for these SOMA candidates.

Outputs remain ordinary offline GLB animation assets, with existing explicit
mapped-humanoid retargeting available after canonical validation. No automatic
bone discovery, skinning, non-humanoid retargeting or deterministic match-simulation
claim follows. No game simulator or robot tracking policy is needed.

## Text reuse, history and constraints

[load_model.py](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/load_model.py)
names the same LLM2Vec Meta-Llama-3-8B-Instruct MNTP base and supervised adapter
family used by native Kimodo. Local text reuse is promising; it is not proven by
4096 dimensions. Validate exact weight revisions, merged adapter, bidirectional
attention, tokenizer/BOS/EOS, prompt formatting, truncation, mean pooling and
normalization against the [vendored encoder](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/llm2vec/llm2vec.py).
Its wrapper encodes batch size 1 and emits [B,1,4096]. Tide's native encoder must
be exposed through a supported local API and parity-tested; do not call ARDY's
optional Gradio text server or substitute another embedding model. Existing
Meta Llama 3 terms/attribution and adapter terms must be retained, and any newly
required revision pinned. No new text weights were downloaded in this spike.

[Ardy state loop](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/model/ardy_model.py)
retains hybrid history, accumulated global translation, current first heading,
frame/token validity masks, constraints indexed in clip time, prompt embedding
and RNG/schedule state. Each chunk creates fresh generation noise, preserves
history while denoising generation slots, appends clean tokens, recentres on the
last history frame, requantizes body tokens and accumulates the translation back.
Crop history and update first heading at the crop boundary. Decoder context and
cropping are also part of seam behavior. This is not merely a transformer KV
cache. Imported history needs the tokenizer encoder; the upstream offline path
currently asserts against combining imported history with history cropping.

The [offline script](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/scripts/generate.py)
defaults to the longest history fitting a trained 10-second window: 240 history
frames plus 60 generated frames for SOMA. History is a multiple of four. A 64-token
export is only 256 frames and cannot hold that 300-frame budget; a full history
requires at least 75 tokens before accommodating a chosen future budget. Demo
and offline defaults differ; do not infer unlimited model context from unlimited
output length. Bound allocation and output length in the native host.

Constraints are observed-motion values plus per-feature masks, with full-body,
end-effector position/rotation, heading and root waypoint/path options. Future
constraints can extend beyond the generated chunk; crop and rebase them with the
history/global origin, preserve masks at token boundaries, and use the same
representation as the checkpoint. See [constraints.py](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/ardy/constraints.py).
No evaluation clips are required: use owner-authored poses/paths. Optional upstream
MotionCorrection is separate; this spike neither ports it nor touches Tide contact
cleanup. Report constraint error before any later correction.

## Native deployment choices

1. **Preferred initial parity path: ORT C++ CPU**, using Tide's already pinned
   1.31.1 MIT runtime, with a native host for DDIM, history, masks and decoding.
   The [C++ API](https://onnxruntime.ai/docs/get-started/with-cpp.html) exists;
   that does not establish ARDY operator coverage or usable speed. A developer-only
   conversion step could be isolated from the delivered tool if separately agreed;
   no Python inference or Python installation may enter the delivered product.
2. **GGML C++ port**, reusing Tide's pinned GGML and text implementation, offers
   the closest fit to Vulkan/CPU deployment. Needs two-stage attention, tokenizer
   encoder/decoder, quantization and state-loop implementation with reference
   fixtures; likely the largest engineering effort. No complete coverage claim.
3. **TensorRT C++** can consume converted networks and run a native host, but adds
   NVIDIA/CUDA runtime and redistribution terms and platform/version constraints.
   Upstream pins `tensorrt-cu12==10.13.0.35`; do not assume its Python package pin
   proves Windows C++ packaging. Upstream warns engine plans require matching TRT
   versions. Treat engines as derived, version/device-specific caches, not portable
   model downloads. Linux-first upstream validation and macOS/AMD exclusion make
   this a possible optional accelerator, not the standalone default.

Upstream [ONNX export](https://github.com/nv-tlabs/ardy/blob/e706e27acc88070918308a10922b8f7174d0825e/scripts/export_onnx.py)
uses opset 17 and exports separated-CFG denoiser and decoder, not the complete
sampler/state/text/history-encoder path. It traces at a fixed maximum token count
and pads at runtime despite dynamic annotations. Decoder outputs are a dict with
separate root/body outputs. Its verification logs warnings and continues even on
failure; export success is not a release gate. No ONNX export was run here.
LibTorch would carry much of PyTorch's native footprint while still requiring a
rewritten host; it is a poor first fit for minimal dependencies.

## Native evidence and reproduction

The standalone [probe](spikes/ardy/probe.cpp) is excluded from Tide's main CMake,
install and release paths. It bounds/parses an F32 safetensors header, checks tensor
shape/byte extents and contiguous nonoverlapping payload, seeks actual projection
weights, checks finite values and executes the learned global-root hybrid input
projection on a deterministic synthetic 148-value vector. This is real checkpoint
loading and learned affine arithmetic, **not inference of an animation**. It does
not read/validate all weight values or execute attention, FSQ or a sampler. The
double-accumulation comparison checks arithmetic only, not upstream parity.

Validated Windows x64 Release, VS 2022 ClangCL 18.1.8, /W4 /WX, TIDE_HIDDEN=1:

- Actual denoiser: 627,156,264 bytes; SHA-256
  `532ead2c40f27733556c099e8d77e8f227e4047b5a88d556b995d0b9c012dec7`.
- Container: 224 F32 tensor entries, 156,781,716 elements (not all necessarily
  trainable parameters); header 29,392 bytes.
- Projection: 1024×148 plus bias; output sum `0.715780863888`, maximum absolute
  difference from double accumulation `4.42014425062e-7`.
- Oversized header, truncated checkpoint and missing-argument refusals passed.
- No GPU inference, GPU timing, GLB generation or Tide preview was performed.
  Thus no GPU mutex contention or engine executable was involved. Future GPU
  validation must acquire Windows named mutex `TideMotionGpuValidation` for the
  entire timed run and release in finally, handling abandoned ownership.

Reproduce from repository root (weights stay in ignored `.research/ardy/`):

```powershell
$env:TIDE_HIDDEN = '1'
cmake -S notes/spikes/ardy -B build-ardy -G 'Visual Studio 17 2022' -A x64 -T ClangCL
cmake --build build-ardy --config Release
# Review the pinned model LICENSE above before obtaining weights.
curl.exe --fail --location --proto '=https' --proto-redir '=https' --max-time 180 `
  'https://huggingface.co/nvidia/ARDY-SOMA-RP-30FPS-KimodoRep-Horizon60/resolve/e6c6804fb6b55636ad927145d74959fd46882ccd/denoiser.safetensors?download=true' `
  -o .research/ardy/denoiser.safetensors
./notes/spikes/ardy/validate.ps1 -Checkpoint .research/ardy/denoiser.safetensors
```

Create the ignored download directory first if absent. The validation script
checks the full SHA-256 before running. Downloaded tokenizer was unnecessary:
its upstream manifest lists 142,218,576 bytes, SHA-256
`c156290ca9da2f821d7e1570aad89f153b943e4bdabb4451eef92ba3fb385398`.
Combined weights are about 769 MB before statistics/text; that is storage,
not a measured RAM or VRAM requirement. Source and weights stayed in ignored
scratch. Only this original probe, instructions and assessment enter Git.

## Value, limits and staged go/no-go gates

Tide's existing Kimodo makes prompt/pose-constrained clips and prompt sequences;
MotionBricks joins clips. ARDY's possible additional value is repeated continuation
with changing prompts and far-future goals, while retaining learned history over
long authored sequences. It does not replace the existing transition workflow.

Upstream's pinned README reports overview/content R@3 of 75.57 for SOMA KimodoRep
versus 79.17 for Kimodo v1.2 on the smooth-root suite, while full-body position
error with text is 2.81 versus 2.93 cm. Those are upstream benchmark results,
not this machine's results or a comparison to Tide's v1.1 runtime. They suggest
a control/semantics tradeoff, not universal quality superiority. No benchmark
data was obtained. The README's protocol uses batch 32 and FP32 text, unlike
the released Kimodo protocol's batch 1/BF16 text; compare within its matched
rows, not directly against Tide's local timings. The paper's 33 ms result does
not establish native SOMA speed.
Existing Tide notes measure about 1.5 seconds per warm Kimodo variant on RTX 4080
SUPER at 50 steps; ARDY has no comparable local generation measurement here.
Cold loading/text latency, peak memory, prompt accuracy, contact slide and seam
quality remain unmeasured. Fewer diffusion steps alone do not prove faster total
authoring, especially with repeated tokenizer and two-stage work.

Implementation and validation are now recorded in [ardy-native.md](ardy-native.md). The former prototype-budget and speed-multiplier gates are withdrawn; measured tradeoffs, not a predetermined speed threshold, determine how the optional backend is described.
