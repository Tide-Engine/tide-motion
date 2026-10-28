# Props and first-person animation investigation

Research date: 2026-10-28. Baseline: dev 8d535b7. This is a source and implementation
audit, not a benchmark of external models. No new weights, evaluation motions,
SMPL/MANO assets or third-party meshes were downloaded. Candidate performance
claims belong to their authors and are not Tide measurements.

Follow-up: [pinned Dex2HOI source/input/operator audit](dex2hoi-feasibility.md).
Tide now implements first-person bone/root cameras, arms-only mannequin
presentation, reusable camera/equipment setups and geometry-aware prop framing.
The baseline limitations below are retained as the audit trail; these four are
resolved by the follow-up, while grip solving and object/bow export remain open.

## Findings and recommended direction

The current body generators can suggest weapon gestures, but Tide does not pass
object geometry, object trajectories or contact schedules into them. A prompt
mentioning a sword is not a sword interaction contract. Better phrasing alone
cannot guarantee grip alignment, collision avoidance or bow-string release.

Keep native body generation. Add explicit equipment/contact authoring and review
first, then evaluate an interaction model against that same contract. Dex2HOI is
the most directly relevant two-object lead found; Text2HOI is a useful hands/object
reference. Neither is a proven sword/bow solution. Full-body generation followed
by camera-aware review is a practical first-person experiment; arms-only output
is a separate export/rendering concern from arms-only model inference.

## Primary-source candidate inventory

| Candidate | Actual task and release evidence | Fit and unresolved work |
| --- | --- | --- |
| [Text2HOI](https://junukcha.github.io/project/text2hoi/) ([code](https://github.com/JunukCha/Text2HOI)) | Text + canonical object mesh/scale to hand/object contact map, motion and learned hand refinement. Released checkpoints linked. | Close-up hand manipulation lead, not a shoulder/elbow/full-arm FPS system. MIT source; separate checkpoint and MANO/artifact terms need inspection. |
| [DiffH2O](https://github.com/facebookresearch/diffh2o) | Text to dexterous hand/object motion, with grasp/interaction stages; checkpoint download script released. | Technical reference; repository explicitly declares CC-BY-NC code. Do not add to commercial runtime under that grant. Other artifacts have separate terms. |
| [OMOMO](https://github.com/lijiaman/omomo_release) | Object-motion-guided hand positions and full-body synthesis. | Useful with an authored object trajectory; not simply text to equipped fighter. Public Python/PyTorch implementation needs body-model assets; MIT source does not settle every artifact. |
| [CHOIS](https://lijiaman.github.io/projects/chois/) ([code](https://github.com/lijiaman/chois_release)) | Initial human/object state, text and sparse object waypoints to synchronized human/object motion with contact guidance. Released model links. | Object-path authoring lead. Public setup needs PyTorch3D, BPS and SMPL-X. Native guided sampling and rig adaptation remain work. |
| [HOIDiNi](https://hoidini.github.io/) ([code](https://github.com/hoidini/HOIDiNi)) | Text, geometry and scene constraints to contact pairs, fingers, body and object motion through two-phase diffusion noise optimization. | Native integration requires backward/optimization support, not just ONNX forward calls. MIT code. Public download combines models with GRAB/body/evaluation assets: do not bulk-import into Tide. |
| [Dex2HOI](https://arxiv.org/abs/2605.30444v2) ([code/checkpoint links](https://github.com/cpratikaki/Dex2HOI)) | Text to single/two-object interactions with coordinated diffusion streams; GRAB and HUMOTO variants. | Priority two-object feasibility audit. Code and pretrained links are now public, despite older search excerpts saying release pending. Python/CUDA/PyG/PyTorch3D/BPS and Blender tooling. Root LICENSE URL returned 404; checkpoint/artifact grants not established here. No weapon coverage/native performance verified. |
| [MAD-HOI](https://arxiv.org/html/2608.10162v3) | Known articulated geometry and text to hand/object motion; completion, infilling and learned termination. | Relevant to staged manipulation. October paper says code will be made public; no usable release established here. |
| [TSHaMo](https://arxiv.org/abs/2603.24407) | Teacher/student framework for text-only hand generation, evaluated on GRAB/H2O. | Hands-only text generation exists; this does not establish full forearm/elbow motion, camera framing or weapon contact. Release/artifact audit pending. |
| [EgoAllo](https://egoallo.github.io/) | Headset SLAM poses and images to estimated wearer body/hands. | Reconstruction from observations, not a text-to-FPS-weapon generator. |
| [EgoMotion](https://arxiv.org/abs/2604.19105) | First-person observations + instructions to 3D human motion through reasoning and diffusion. | Egocentric conditioning research; not equivalent to isolated game viewmodel arms. |

No candidate above was run here. Novel object shape generalization is not evidence
of unseen combat skill generalization. None of the reviewed evidence establishes
reliable sword/shield fighting or bow draw/release with Tide rigs. Before any
integration, pin source and artifact revisions, inspect actual grants/notices,
inventory operators and test native parity. Do not ship research-only checkpoints
or evaluation motion data. No model or dependency pins changed in this work.

## Tool audit: limitations independent of the body models

Evidence: src/main.cpp preview parsing/validation; src/platform.cpp body_scene;
src/motion.hpp constraints/PreviewOptions; src/soma77.cpp; src/gltf.cpp;
notes/ardy-native.md, notes/dart-backend.md and PROMPTING.md.

| Limitation | Source | Consequence and improvement |
| --- | --- | --- |
| Body pose/effectors/root-path inputs, no object geometry/contact schedule | Tide interface and backend representations | Add object-space grip targets and timed contacts with measured residuals. |
| One preview prop on baseline | Tide preview | Sword/shield cannot be reviewed together. Fixed here with repeated independent attachments. |
| Attachments are preview-only | Tide asset/export path | Generated GLB does not contain attached meshes/object tracks. Design export with object nodes and retained equipment recipes. |
| Rigid attachment has no second-hand constraint | Missing tool contract and model conditioning | Two-handed sword needs an object-local secondary grip and wrist position/orientation measurements over a contact interval. |
| SOMA-30 lacks fingers; SOMA-77 expansion supplies relaxed/source hands | Representation and backend motion quality | More bones do not imply inferred gripping fingers. Mannequin is SOMA-30; detailed review needs a compatible rig/mesh. |
| Body retargeting does not author arbitrary equipment tracks | Tool mapping/export | Preview can follow character grip nodes, but retargeting does not invent independent object motion. |
| Framing uses body joint extents, not weapon bounds | Tide preview | Long weapons can leave the image. Include transformed prop bounds. |
| Fixed paired third-person front/profile camera | Tide preview | Add explicit camera basis, FOV, offset and near plane for bounded first-person review. |
| No validated arms-only mesh/export workflow | Tide tooling/content | Preserve ancestor transforms or bake before pruning; use supplied skinned arms meshes. No automatic skinning claim. |
| No bow/string/arrow animation and release-event contract | Tool assets/events and training coverage | A rigid bow cannot show string tension, arrow nocking or release. Add separate articulated/object tracks and events. |
| No continuous grip/penetration or camera-space quality report | Tide diagnostics | Body/foot metrics cannot certify equipment/FPS quality. Add contact drift, orientation error, separation and screen occupancy. |

## First-person route using full-body inference

Recommended experiment, not yet implemented or quality-validated:

1. Generate full-body shoulder/elbow/wrist motion with explicit handedness.
   Select Kimodo by action per PROMPTING.md; compare matched settings. A tested
   sword-worded kick does not establish sword-swing quality.
2. Retain the rig and place a review camera at an authored eye transform.
   Compare body-following and stable gameplay-style views. Animated head bob/roll
   may be undesirable; a head bone is not automatically a good camera rig.
3. Render a supplied arms mesh or hide unwanted mesh regions while preserving
   shoulder ancestors. Deleting spine/root channels changes wrist trajectories.
   Partial humanoid mapping is not a validated arms-only export contract.
4. Judge weapon silhouette, visible hands, elbow plausibility, near clipping and
   penetration throughout the clip at the intended aspect ratio/FOV.
5. Constrain/regenerate from selected body poses where possible. Any future
   offline grip correction must be identified/measured separately from learned
   inference; it is not Tide deterministic match simulation.
6. Bake approved camera-relative arms motion and object tracks to ordinary GLB,
   retaining the rig mapping and equipment/camera recipe.

Hands-only models still require wrist-to-arm integration and shoulder/elbow
behavior. Camera changes cannot repair contacts, and good contacts alone do not
guarantee an attractive FPS pose.

## Next experiments and acceptance criteria

Test one-hand sword, sword + shield, two-hand sword, then bow. These isolate one
persistent grip, two separate grips, two grips on one object, then changing contacts
and articulation/release. Use owner-provided licensed meshes and own reference
clips, without bundling evaluation motions.

Keep at least eight matched seeds per action and exact recipes. Inspect complete
bounded front/profile and first-person recordings. Record grip translation and
orientation residuals over contact intervals, second-hand drift, maximum temporal
jumps, penetrations, near clipping and hand/weapon visibility. For bows also check
nock alignment and contact/release event timing. Count failed actions before ranking
contact metrics: a still character is not a successful swing. Choose thresholds for
the target rig/camera before ranking; no new quality pass or superiority is claimed.

Engineering order: multiple prop preview (this change), reproducible equipment and
camera recipes/first-person review, grip diagnostics/object-space constraints,
object-track export, then native interaction-model feasibility (Dex2HOI two-object
and Text2HOI hands/object are initial candidates). Keep learned motion inference;
do not replace it with procedural animation or blending.
