# Dex2HOI native feasibility audit

2026-10-28. Source pinned to
[`a92aca6b48b7a0c9a45a1465aa984213d3c8ceda`](https://github.com/cpratikaki/Dex2HOI/tree/a92aca6b48b7a0c9a45a1465aa984213d3c8ceda).
Inspected source locally in ignored `.research/dex2hoi/source`; no upstream code
was copied into Tide. No checkpoint, motion dataset, body-model asset or evaluator
was installed. This is an operator/input audit, not inference or parity evidence.

## What is promising

The [paper](https://arxiv.org/abs/2605.30444v2) specifically models one or two
objects, with coordinated diffusion streams and fusion. That matches the structure
of sword + shield better than a body-only generator. Its contact mechanism is a
learned prior, not an exact rigid-grip constraint. Weapon skills and bows still
require independent evidence; household manipulation quality does not establish
combat quality. The paper's speed results are not native Tide measurements.

## Released implementation, rather than a text-only demo

The pinned [HUMOTO runner](https://github.com/cpratikaki/Dex2HOI/blob/a92aca6b48b7a0c9a45a1465aa984213d3c8ceda/model/hoi/humoto_inference.py)
constructs a dataset-backed loader and takes motion prefixes, initial root/object
state, geometry and text from it. Defaults include 10 context frames, 60 predicted
frames, two object slots, 128 object points and 30 fps. These are source defaults,
not verified checkpoint configuration. `args.json` is loaded from the training
directory. Normalization requires `mean.pt`, `std.pt` and `eps.pt` there.

The public README links HUMOTO/GRAB checkpoints and evaluators in one Drive folder.
The web tool could not open that folder; a direct public HTML fetch did not yield
an artifact listing. Actual filenames, sizes, hashes, packaged configuration and
artifact-specific grants remain unverified. Do not assume `model.pt` alone is
enough or require importing evaluation sequences to build a Tide authoring UI.
The future standalone input path should use owner-authored history and geometry.

## Native execution inventory

| Stage | Pinned implementation evidence | Native work |
| --- | --- | --- |
| Text | `model/hoi/hoi_model.py`, `hoi_motion_fusion.py`: frozen CLIP ViT-B/32, 512-dimensional embeddings, OpenAI CLIP tokenization | Tide already has native CLIP infrastructure for other backends; verify exact tokenizer, weights, projection and numerical parity before reuse. |
| Object geometry | `object_encoder_pointwise.py` and `object_encoder_global.py`: PointNet-style layers, farthest-point sampling, radius neighbors, pooling and k-NN interpolation through PyG/torch-cluster | Export dense learned portions; implement or convert neighborhood/sampling operations faithfully. Object features are cached before the sampling loop, so geometry work need not repeat every DDIM step. |
| Optional BPS | Constructor selects PointNet or `ObjectBPSEncoder` by configuration | Audit actual checkpoint selection. A BPS branch is not permission to substitute it for a PointNet-trained model. |
| Motion networks | Dual decoder streams, self/cross attention, layer normalization, MLP/transformer fusion, configurable masks and initial-state conditions | Forward graph export appears plausible; export and native output parity have not been executed. |
| Sampling | `humoto_inference.py`, `inference_utils.py`: classifier-free guidance, DDIM and autoregressive prefix windows | Native host owns RNG/noise, schedule, masks, history, chunk placement and exact output lengths. This forward inference path does not inherently require HOIDiNi-style noise optimization/backpropagation. |
| Normalization | `normalizer.py`, `humoto_dataset_multiobj.py` | Pin statistics/configuration with network artifacts and validate finite dimensions and nonzero scales. |
| Human representation | `human_model/bone_names.py`, `humoto_fk_wrapper.py`: explicit Mixamo-name hierarchy with fingers | Map explicitly to SOMA; validate rest geometry, units, root transforms and wrist FK. Preserve finger tracks rather than pretending SOMA-30 predicts them. No automatic bone identification or skinning. |
| Objects | `obj_slot_utils.py`: legacy 9D or 18D wrist-mixture slots | Decode exact checkpoint representation. The 18D translation combines left/right wrist-relative positions and free translation with softmax weights; object rotations remain global 6D. These are not rigid attachments or two-hand IK. |
| Output | HUMOTO runner saves tensor results/body/object poses; optional FBX/Blender path | Native GLB body + independent object tracks and an equipment recipe. Blender and Python must stay out of the shipped runtime. |

No native operator coverage or speed claim follows solely from this inventory.
Tide's existing ONNX Runtime is useful infrastructure, not a completed backend.
Keep engine and authoring CLI independent; inference is offline model arithmetic.

## Release terms and scope

The pinned tree has no top-level LICENSE/NOTICE/COPYING file. It contains an MIT
license for the vendored diffusion planner and a PyTorch3D license in that subtree;
those do not establish a grant for every new Dex2HOI file or released checkpoint.
This is an unresolved artifact finding, not a conclusion that every use is
forbidden. No model files enter Git or packages under an assumed grant. Preserve
actual upstream terms if integration proceeds. Do not redistribute research-only
checkpoints, body assets or evaluation motion data.

## Concrete feasibility sequence

1. Obtain an exact model/config/statistics inventory and actual terms. Pin hashes;
   isolate model artifacts from evaluators, body meshes and dataset sequences.
2. Build an owner-authored Mixamo-compatible initial history and two licensed prop
   meshes with explicit units/origins. Prepare the actual point features and text.
3. Developer-only export/reference tools compare text, object encoder and motion
   networks to native execution on fixed synthetic/owner-authored inputs. Ship
   neither Python nor the reference pipeline.
4. Implement native DDIM, prefix normalization/decoding and 18D object FK if used.
   Verify each stage, then multi-chunk body/object GLB round trips and seed recipes.
5. Compare sword, shield+sword, two-hand sword and bow against current backends
   with continuous grip, timing and first-person checks. Reject wrong actions
   before comparing smoothness. Retain selected assets and reproducible recipes.

The source audit supports a focused native feasibility project, not immediate
integration or a promised quality improvement. Tide-side review limitations are
being fixed independently: first-person cameras, arms-only mannequin presentation,
reusable setups and geometry-aware prop framing now have native implementations.
Continuous grip solving, independent object-track export and articulated bow/event
authoring remain separate substantial features, not hidden options of this model.
