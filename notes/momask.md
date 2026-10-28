# MoMask native investigation and implemented representation study

Reviewed 2026-10-28. **No supported MoMask backend is shipped.** The external gate
is the missing checkpoint-specific commercial grant, not an ONNX operator failure.
Native source work and tests below are implemented; they do not generate motion
from text. Existing backends, defaults, install manifests and recipes are unchanged.

## Exact release and rights evidence

Primary source: [EricGuo5513/momask-codes](https://github.com/EricGuo5513/momask-codes),
revision `94a6636c9c463b7a9414c3401a6f1b67e6c51824`. Its README licenses **code**
under MIT and distinguishes third-party libraries and dataset terms. The retained
notice is `licenses/MOMASK-SOURCE.txt`. `momask-source-lock.json` pins the URLs and
SHA-256 of 52 reviewed source/documentation files; it is not a model installer.

The [author project page](https://ericguo5513.github.io/momask/) links this repository
and [MeYourHint/MoMask](https://huggingface.co/spaces/MeYourHint/MoMask). The latter
at `4b0323058afeb8d4443d5b370a9e35fd9537222b` has the same MIT software license,
a UI configuration README, and a download script pointing at the **same archive**.
Neither adds an explicit checkpoint or generated-output grant. GitHub's releases
API returned an empty list; issue searches for `license` and `commercial` returned
zero results. These searches establish what was found, not a universal negative.
No person was contacted, and no research-only agreement was accepted.

The exact HumanML3D archive is
[humanml3d_models.zip](https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view),
as linked by the pinned official `prepare/download_models.sh`. Bounded HTTP Range
requests inspected its ZIP directory and extracted only three small `opt.txt`
configuration texts, without extracting/loading any checkpoint or statistics.
Archive size: **188,067,025 bytes**; directory: offset **188,064,796**, size **2,207**,
SHA-256 `5073f02c987cddfb93c66eecc134463e483ad22d4f6f8259198eecc191b0a9ac`.
All 18 directory entries and the configuration texts/digests are recorded in
`momask-archive-inventory.json` and `momask-configs.json`. There is **no license,
README, or other grant file in that directory**; none of the configuration texts
contains licensing terms. Checkpoint payload metadata has not been unpickled.
The directory hash is not a hash of the complete archive or its checkpoints.

Necessary release artifacts, relative to `checkpoints/t2m`:

| Component | Exact release artifact | Evidence / remaining gate |
| --- | --- | --- |
| RVQ tokenizer/decoder | `rvq_nq6_dc512_nc512_noshare_qdp0.2/model/net_best_fid.tar` | 84,070,519 uncompressed bytes; commercial checkpoint grant missing |
| Masked transformer | `t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns/model/latest.tar` | 55,933,307 bytes; commercial checkpoint grant missing |
| Residual transformer | `tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw/model/net_best_fid.tar` | 61,196,511 bytes; commercial checkpoint grant missing |
| Normalization | RVQ `meta/mean.npy`, `meta/std.npy` | 2,232 bytes each; exact content/provenance/redistribution grant and content hashes still needed |
| Architecture options | Each of the above three directories' `opt.txt` | Individually hashed metadata, inspected; no license terms |
| Optional auto length | `length_estimator/model/finest.tar` | 1,745,581 bytes; omit from first runtime by requiring explicit duration; grant missing if used |
| Text | OpenAI CLIP ViT-B/32 + BPE | Separate required inference component; see below |

No full checkpoint hashes are asserted: no checkpoint was integrated or installed.
A mutable Drive ID plus archive-directory hash is insufficient for Tide's model
installer. Once rights are established, pin complete archive and individual input
and converted output hashes, sizes, conversion tooling and exact license texts.
The KIT archive is a different model/representation and is outside this deliverable.

[HumanML3D](https://github.com/EricGuo5513/HumanML3D/tree/9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23)
has MIT software, but identifies AMASS as its motion source and requires separate
SMPL+H/DMPL assets to reproduce the dataset. The
[AMASS primary license](https://amass.is.tue.mpg.de/license.html) restricts dataset
use and commercial training. **That dataset restriction alone is not a conclusive
decision about a separately licensed pretrained model or its outputs.** It makes
provenance and the scope/authority of a checkpoint grant important; it cannot be
resolved by a repository's software license or another uploader's license badge.
The observed license-page hash/date are recorded in the source lock.

Concrete missing evidence: an author/rightsholder release term covering commercial
offline inference, conversion and use of outputs for these three exact checkpoints
and their statistics, plus evidence that necessary underlying rights are cleared.
Does that grant permit distributing converted weights, or only owner downloads?
Does it cover the normalization arrays? Does any restriction attach to generated
outputs? Alternatively, an officially cleared, compatible RVQ+masked+residual set
trained on licensed motion with corresponding statistics/configuration would work.
No such primary release was found in the official artifacts above. This is
**commercial eligibility not established**, rather than a claim that all MoMask
outputs are categorically prohibited. A third-party repack does not supply the
missing original-rightsholder evidence.

CLIP source audited at `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (MIT).
Its official `clip/clip.py` pins the ViT-B/32 file at
`https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`
with expected SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The official BPE file is `clip/bpe_simple_vocab_16e6.txt.gz` at that source revision.
Neither asset was downloaded for this study; BPE content hash, exact distribution
notice scope and native tokenizer fidelity remain release checks. CLIP's model
card discusses deployment limitations; that discussion is not treated as a
research-only license. Do not substitute OpenCLIP, another embedding dimension,
or normalized CLIP embeddings without parity evidence.

## Necessary inference components and native route

**SMPL, SMPL-X, DMPL, PyTorch3D, body meshes, licensed body-model pickles, the
upstream BVH template and evaluation motion are unnecessary for skeleton-only
inference.** The necessary network calls in `gen_t2m.py` and `edit_t2m.py` produce
263-dimensional feature rows. `recover_from_ric` uses only row arithmetic; native
rotation reconstruction and the existing GLB writer replace visualization and
BVH/mesh fitting. `Joint2BVHConvertor`, its template/foot IK, plotting, ffmpeg,
evaluation models, GloVe and dataset loaders must be excluded, not installed
because they appear in upstream requirements. `process_file` is not standalone:
it depends on module globals including target offsets initialized from an example
dataset skeleton. A native encoder must use an explicitly authorized skeleton
calibration, never fetch upstream example `000021` or test motion `000612`.

Release options inspected directly: RVQ has six independent 512-entry, 512-wide
codebooks, two stride-2 encoder levels (4 source frames/token), width 512, depth 3,
dilation growth 3, ReLU and no normalization. Decoder sums selected codebook
vectors, then residual convolutions, nearest upsampling and convolutions. In eval,
the encoder uses residual nearest-code quantization with no training dropout or
EMA/codebook updates. Omitted quantizers are padded with -1 and zero their vectors.
The current export fixture fixes six quantizers; it does not prove optional coarse
decoding with a different quantizer-axis length.

Both transformers have eight layers, width 384, FF width 1024, six heads and
dropout 0.2 (disabled for inference). The masked model embeds code IDs and adds
sinusoidal positions, prepends projected CLIP text, applies bidirectional attention
with padding, and emits 512 code logits. Mask/pad IDs are 512/513. Residual steps
consume accumulated previous-code embeddings, prepend text and quantizer-index
conditioning, and project using quantizer-specific tied weights. Its pad ID is
512 and its projection includes 513 logits. Preserve that upstream behavior;
do not silently remove the pad logit. `process_embed_proj_weight` must run with
frozen weights or produce registered graph parameters before export.

CLIP is tokenized with truncation to 77 positions (including start/end), byte BPE,
HTML unescaping, ftfy cleanup, whitespace/lowercase and Unicode regex. The output
is the unnormalized, float32 512-vector selected at the end token. A C++ tokenizer
must test ASCII, Unicode, entities, malformed input and truncation. These text
tests and CLIP numerical parity are **not attempted yet** because this deliverable
stops before licensed checkpoint integration. They are not known operator blockers.

Choose existing **native ONNX Runtime 1.31.1**, with C++ orchestration, rather than
a new hand-written transformer/conv runtime or LibTorch dependency. Export one
text graph, encoder/quantizer, decoder, masked-step logits and residual-step logits;
keep loops, masks, CFG, code history, top-k and RNG in C++. Do not export the whole
Python sampling loop or ship a Python process. New runtime support would reuse
owned ORT sessions and a single text embedding per prompt in the run context.
No sessions/model-install/CLI/batch controls are exposed by this study.

## Implemented operator proof and technical findings

`tests/momask-export.py` runs **CPU random-weight architecture fixtures**, matching
the inspected options, without any pretrained weights or motion data. Constructors
are prevented from loading CLIP; arbitrary 512-dimensional tensors supply text.
This omission is explicit, and these tensors are never presented as text inference.
The developer environment is Python 3.11.15 with the complete version/hash lock in
`momask-export-requirements.txt` (Torch 2.7.1+cpu, ONNX 1.18.0, reference ORT 1.22.0).
Torch/NumPy are BSD licensed, ONNX Apache-2.0, ORT/einops MIT, tqdm MIT/MPL-2.0;
none of these developer packages or generated ONNX files ships in the runtime.

The first legacy opset-17 export froze attention's reshape to the traced sequence
length and failed at other lengths. Fixed by an explicit dynamic self-attention
decomposition using Linear, reshape/transpose, scaled MatMul, padding and Softmax.
It is tested against the original PyTorch implementation before export (max error
7.15e-7 masked / 3.73e-7 residual). Tied residual projection initially appeared as
an unregistered gradient-bearing constant: freezing parameters before assembling
the tied projections fixes export. No upstream source file is modified.

Exports pass ONNX checking and reference CPU parity. The separate native
`momask-ort-tests` executable loads the exported graphs using Tide's pinned
**C++ ORT 1.31.1**, reuses one session per component across 4/7/49-token cases,
and compares every output to saved PyTorch tensors. All 12 cases pass:
masked logits max absolute error 1.19209e-6; residual logits 6.85453e-7;
decoder 3.81842e-7; encoder integer IDs exact. Float bounds are
`abs(error) <= 3e-5 + 3e-4*abs(reference)`. Exact exported graph hashes, operator
inventories and reference measurements are in `momask-export-results.json`.
Graphs and all binary fixtures remain ignored local artifacts.

One CPU observation (four intra-op threads): session loads 24–114 ms; 49-token
masked step 3.89 ms, residual step 5.94 ms, RVQ decode 5.83 ms, encode 5.95 ms.
These are random-weight component smoke timings, **not text-to-motion latency,
memory/quality measurements or a comparison against Kimodo**. No GPU work ran;
future inference/benchmarks must acquire `TideMotionGpuValidation`. No pretrained
per-component parity, native CLIP parity, complete sampler/RNG parity, real
generation, real inpainting, visual quality or Kimodo comparison is claimed.

## Native motion representation and implemented conversion contracts

`src/humanml.*` is compiled into the model-free test executable only. It implements
finite/positive normalization checks, normalization roundtrip, exact upstream
root/RIC recovery, incoming-edge rotation FK, a lossy RIC-to-canonical-SOMA study,
30 Hz resampling, endpoint-checked splice and the cosine remasking count.
It is not a fallback motion generator and is unreachable from the shipped CLI.

HumanML feature layout (zero-based, end exclusive): `[0,4)` = root angular
increment, planar displacement x/z and absolute root height; `[4,67)` = 21
heading-local positions; `[67,193)` = 21 incoming-edge continuous 6D rotations;
`[193,259)` = 22 local velocities; `[259,263)` = four foot-contact flags.
Statistics apply to all 263 channels. Positions are Y-up metres, canonicalized
to initial +Z with ground/origin removed upstream. Heading scalar integrates
**half angles**, not radians of yaw/second: at frame f sum angular increments
through f-1, form WXYZ `(cos(a),0,sin(a),0)`. Previous x/z displacement rotates
by its inverse at the current frame before accumulation. Root height and RIC
joint y values are absolute; only RIC x/z get root translation. Tide stores XYZW.
Save initial world origin/heading separately for editing and restore them later.

The 6D block uses the first two matrix columns and cross-product
orthonormalization. Degenerate/nonfinite columns are refused. Upstream FK rotates
each incoming offset by the accumulated rotation **including the child's own
rotation**, unlike glTF parent-local transforms. Its arm chains reset accumulated
rotation at the root although collars attach to spine3. The native FK study
preserves and tests this convention. `recover_from_rot` and `recover_from_ric`
are separate upstream paths and need not agree for independently predicted
redundant features; treating the 6D block as SOMA local quaternions is incorrect.

Explicit HumanML index -> SOMA name correspondence:

| HumanML indices | SOMA names, in the same order |
| --- | --- |
| 0,1,2,3,4,5 | Hips, LeftLeg, RightLeg, Spine1, LeftShin, RightShin |
| 6,7,8,9,10,11 | Spine2, LeftFoot, RightFoot, Chest, LeftToeBase, RightToeBase |
| 12,13,14,15 | Neck2, LeftShoulder, RightShoulder, Head |
| 16,17,18,19,20,21 | LeftArm, RightArm, LeftForeArm, RightForeArm, LeftHand, RightHand |

The implemented conversion uses RIC primary segment directions and decoded 6D
edge frames' Y axes as roll hints. It fits these to canonical rest segment frames,
then derives parent-local rotations; canonical translations/proportions remain
fixed. Root height/travel remain metres without guessed leg scaling, floor
correction or recentering. Head, toe and hand leaves inherit parent rotation;
Neck1, jaw, eyes and finger endpoints keep local rest rotations. A collinear roll
hint selects a documented orthogonal fallback. These are reconstruction choices,
**not extra predicted fingers or a guarantee of source position/roll fidelity**.
The fitter is tested on neutral and articulated authored geometry. It still needs
real generated turn/twist/grounding evaluation and a defensible calibration to the
exact release's training skeleton proportions. No dataset skeleton is bundled.

20 Hz -> 30 Hz uses root linear interpolation and shortest-sign quaternion slerp,
sampling `f/30` through `floor((N-1)*30/20)`. No fabricated last pose interval is
added; a non-grid final sample can lose less than 1/30 second. This interpolation
is representation resampling, not an inference substitute. The output passes
`require_soma` before ordinary GLB export and before existing mapped-humanoid
retargeting. No automatic rig identification, skinning or non-humanoid support.

Model-free tests use independently authored offsets and analytic trajectories:
normalization roundtrip/refusals, 90-degree half-angle integration, shifted
velocity and absolute RIC height, incoming-edge chain FK and collar reset,
canonical rest/limb directions/fixed lengths, 20->30 frame count/root timing,
GLB export/reload, remask endpoints/monotonicity and exact splicing/refusals.
The optional GLB fixture has seven frames; it is synthetic contract data, not
MoMask inference or evaluation motion. All six CTests passed in a fresh checkout
Windows x64 ClangCL 18.1.8 Release build with TIDE_HIDDEN=1 before rebasing.
After rebasing onto origin/dev 15c78fd, all seven integrated CTests and the native
GLB roundtrip pass with isolated worktree TEMP/TMP. The native ORT checker also
passes all 12 cases after rebuilding with warnings as errors. Runtime staging
includes this report, pin records and notice. CPU smoke timings varied under
concurrent work and are not an isolated benchmark.

## Sampling and editing constraints

Upstream mask generation visits linspace(0,1,steps), remasking the lowest-confidence
`max(1, round(cos(pi*t/2)*token_length))` tokens with ties-to-even rounding. It uses
`ceil((1-topkr)*vocabulary)` retained logits, CFG `u + scale*(c-u)`, constant
temperature (the annealing code is commented), and either categorical or Gumbel
sampling. Scores use unfiltered, untempered logits, then kept tokens score 1e5.
Residual generation samples layers 1..5 with accumulated code embeddings and
Gumbel noise. Generation uses residual CFG 5 while editing uses 2; do not merge
these defaults accidentally. Default masked steps = 18, guidance = 4, topkr = .9,
temperature = 1. `fixseed` seeds Python, NumPy and Torch; equal numeric C++ seeds
do not imply equal Torch categorical/Gumbel streams. A versioned native RNG,
serialized seed/sampler policy and fixed-random-tensor parity tests are still
needed; no cross-platform deterministic match-simulation claim is made.

Upstream edits floor frame ranges into 4-frame tokens and regenerates residual
tokens **over the whole sequence**. Convolution receptive fields and cumulative
root integration can change unmasked decoded frames. Copying base tokens alone
cannot preserve source poses. Native `splice` keeps every original canonical
30 Hz transform outside explicit `[begin,end)` exactly and requires both endpoints
inside the range to match before copying the generated interior. It refuses a
mismatch; it does not fabricate a blend. Rotation tolerance is `1-abs(dot(q1,q2))`,
position tolerance is metres. C0 root/rotation continuity is checked; derivative
continuity and contact quality remain separate evaluation gates.

Future runtime editing must encode owner-owned/canonical motion at 20 Hz with
valid calibrated HumanML statistics, conservatively expand masks to covering
tokens, restore original world placement, and splice only within original frame
ranges. Reject empty/reversed/out-of-bounds or too-short ranges; validate root
and endpoint errors before writing any output. Preserve all source frames beyond
196 rather than padding/truncating silently. First supported runtime should bound
the model window at the published 196 frames and require explicit duration.
Support text, duration, seed, CFG/sampling and temporal masks only after tests;
there is no evidence for Kimodo path/effector/keyframe controls on MoMask.

## Reproduction and remaining release gates

Source-only audit on Windows: `pwsh -File tests/momask-sources.ps1 -FetchMissing`
downloads only pinned source text to ignored `.research/momask-source`; without
the switch it verifies existing bytes. It never runs upstream download scripts.
`python tests/momask-archive.py .research/UNIQUE-metadata` uses stdlib Python and
curl to verify the pinned archive directory and inspect only option texts.
Use a fresh output directory every time. Full archive/checkpoint downloads are
deliberately absent. Package only Tide's existing runtime and notices.

For developer export, create an isolated Python 3.11 environment and install the
hash-locked requirements, using the official PyTorch CPU wheel index for
`torch==2.7.1+cpu`; e.g. `uv pip sync --require-hashes --extra-index-url
https://download.pytorch.org/whl/cpu notes/momask-export-requirements.txt` with that
environment active. Run `python tests/momask-export.py .research/momask-source
.research/UNIQUE-export`. Build the normal CMake project with BUILD_TESTING, then
run `humanml-tests [UNIQUE.glb]`, `momask-ort-tests .research/UNIQUE-export`, and
CTest. Native tests do not need Python. No developer tooling is a user dependency.

**Cannot establish rights yet:** the exact checkpoint/statistics commercial grant
and provenance/authority described above. **Implemented and passed:** native
representation contracts, source-only SOMA fitter/splice, dynamic exports and
native ORT operator parity with random weights. **Not attempted yet:** loading or
converting real checkpoints, native CLIP/tokenizer and full sampler integration,
SOMA->HumanML encoder/calibration, actual generated/edit endpoint solving, session
integration, install/recipe/CLI/batch preflight and real inference/quality/memory
tests. They are work to do after rights clearance, not evidence of impossibility.
Do not advertise a closed capability gap, benchmark speedup or opt-in backend
until those real native end-to-end gates pass.
