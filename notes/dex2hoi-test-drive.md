# Dex2HOI private weapon test drive

2026-10-28. Three native CPU trials using the private HUMOTO EMA bundle, full500
original cosine timesteps, DDIM eta0, CFG2.5, seed7 and four threads. Each asset
contains the ten supplied30Hz prefix frames plus sixty learned frames. No IK,
grip correction, object remounting, snapping or pose blend was applied.

**Result: this configuration is not a usable weapon-animation solution yet.**
Contacts fail badly, the prefix-to-generated transition jumps, and the two matched
text prompts yield nearly identical motion. These are observations on one
manually authored static guard prefix/seed, not a general benchmark or proof that
all possible Dex2HOI inputs fail. Do not conflate native numerical parity with
motion quality or silently repair these assets before judging them.

## Controlled inputs

The previous integration fixture used SOMA identity rest orientations and
approximate object samples. That fixture's arm bases differed from the released
reference, and its point-cloud extents differed from its meshes. It was a plumbing
test, unsuitable for attributing visual failure to the model.

These trials instead use the exact65bone released reference rest frames privately
from pinned source a92aca6b48b7a0c9a45a1465aa984213d3c8ceda. Reference JSON SHA256
`02cfa7e21ec93faa2a25a99b614de0b84ed8daf8dca0927060afd165d13e2a36`.
Only rest matrices were read; no upstream character mesh, skin, evaluation motion
or dataset prefix was used. Those rest matrices, converted fixtures, generated
assets and weights stay outside Git/packages. The test's static guard rotations,
sword/shield geometry, surface samples and visualization capsules are newly authored.
Small rest-matrix nonorthogonality was projected to rotations (max3.36e-7).

Sword geometry is .22x.96x.04m including guard/handle; shield .44x.56x.10m including
back handle. Each object's128 samples lie on its actual mesh surface within1.4e-8m.
Meshes/poses use the same sample-mean local origin. Explicit palm and handle
markers align within2.6e-8m initially. Single-sword conditioning duplicates the
second object slot, with its export disabled, matching the upstream convention.
Own full-body/arms-only capsule mannequins retain the explicit65bone rest rig;
they visualize the generated poses and are not an automatic skinning feature.

Private native bundle manifest:
`51aba4a1a0fb56f56231e26a434dd03d7adbe8a1aba76049a41c2e755c930063`.
Lift/slash input hash:
`502cd9e560299b95906f5bc8e2d37ecac0d945717ce8950d63538ab818784d7d`.
Sword/shield input:
`a20a90c6553d04cf5d4231a23ab12a424ed3fc59b17b2fe3f41efe997f360e6f`.
Recipes verify identical bundle/history/input/seed/schedule/CFG/device/threads for
the matched single-sword comparison. Nine token positions differ, so the prompt
change actually reaches tokenization.

## Raw generated observations

| Trial | Prompt | Mean / max designated grip gap | Object jump at frame9→10 |
| --- | --- | ---: | ---: |
| Sword lift | A person holds a sword in the right hand and slowly lifts it upward. |36.38 /43.81cm |67.71cm |
| Sword slash | A person holds a sword in the right hand and swings it forward in a strong slash. |36.48 /43.74cm |68.19cm |
| Sword + shield: sword | A person holds a shield in the left hand to block and swings a sword forward with the right hand. |44.50 /47.02cm |80.61cm |
| Sword + shield: shield | Same two-prop trial |11.22 /13.79cm |47.15cm |

Gaps compare the explicitly authored palm/handle markers over frames10..69.
They are not distances to the nearest arbitrary part of the sword; touching a
blade is not a correct handle grasp. Mean sword orientation errors are about59°
for the single-prop trials; these are also designated-marker errors.

Lift versus slash differs by only3.38mm body-joint world RMS,3.58mm right-wrist RMS,
9.23mm sword-origin RMS and .62° mean sword orientation. The requested lift's
top point starts at2.017m but stays at1.398..1.904m throughout learned frames,
ending at1.398m. These data and inspected playback do not demonstrate a lift or
a convincing forward slash. No text-adherence score has been invented.

All samples are finite; source prefix positions remain exact with rotation error
≤1.71e-6°. Bone-length variation ≤5e-16m is consistent with the exported rigid
rotation hierarchy, not evidence of good contact/anatomy. Native inference times
were130.865s lift,122.643s slash and127.859s two-prop; early runs overlapped on CPU,
so these are local timings rather than controlled performance comparisons.

## Reproduction and viewing

Private files are retained under `.research/dex2hoi/quality-inputs`,
`quality-tests`, `quality-metrics` and `quality-preview`. The independently written
developer [measurement tool](../tools/dex2hoi/measure.py) reads actual exported
GLB channels/FK. It measures prefix, motion, designated markers and fingertip-bone
to rigid mesh triangle distances. The latter are bone proxies, not skin contact,
penetration or a grasp certificate. Without a marker file it explicitly labels
its frame9-calibrated reference; that fallback is not an authored-contact check.

```powershell
tide-motion dex2hoi generate --input .research/dex2hoi/quality-inputs/sword-only.json `
  --prompt "A person holds a sword in the right hand and swings it forward in a strong slash." `
  --seed 7 --steps 500 --guidance 2.5 --threads 4 -o C:/private/new-slash.glb
python tools/dex2hoi/measure.py C:/private/new-slash.glb `
  .research/dex2hoi/quality-inputs/sword-only.json C:/private/new-slash `
  .research/dex2hoi/quality-inputs/sword-only-grips.json
```

The new native `dex2hoi preview` shows body plus independent learned props in
third person and first person. Plain generic skeleton/character previews did not
provide that combined view; this was a tool gap, now fixed. Explicit node selection
also prevents render-only nonuniform prop transforms from being mistaken for
unsupported skeletal scaling. Source geometry and animations are not rewritten.

Native70frame GIFs and captures were inspected with TIDE_HIDDEN=1, explicit GIF/
frame bounds and held TideMotionGpuValidation, using the staged Tide executable
built from clean engine367ae4da2d31913176ae1e2624e1d78a1204f03c. First-person testing
uses root-carried camera, own arms-only geometry, offset0,.08,.15m, pitch25°, FOV100°,
near .01m. These chosen view settings expose the arms; they do not fix the grip or
make a full-body motion into a dedicated first-person action. The default camera
and a trial moved behind the shoulders respectively lost the hands/caused close
upper-arm occlusion; those camera experiments were kept separate from motion.

The inline diagnostic viewer uses the independently extracted native world
samples, actual prop triangles and explicit grip markers. It offers shared frame
scrubbing across all three trials and a root-carried perspective. It is a skeleton
inspection view, not a claim of engine renderer equivalence.

## Follow-up to owner grip feedback

The owner confirmed that prop positions/grips were wrong. An actual-input check
compares all429 raw prefix features against the pinned upstream encoding helpers:
maximum1.1921e-7, including object slots and finger/surface proximity weights.
Native full-mixture encode/decode reconstructs the two input objects within
1.813e-7m. No position/encoding bug was established for this fixture. The model's
conditioning has geometry/history but no explicit designated handle/palm contract.

A separately labelled authored candidate binds the existing palm/handle markers
through native equipment `space: "grip"` anchors. The original raw GLB is retained.
The native `dex2hoi extract-body` command exposes the original body without prop
nodes for this workflow; no Python runtime or new inference is involved.

Independent exported-file measurements over all70 frames: maximum designated
grip gap3.844e-7m for sword and2.710e-7m for shield (both below.001mm), versus raw
47.02cm/13.79cm. Every body/finger local position is unchanged; rotation difference
is at most2.10e-6degrees from normal GLB quaternion round trips. No IK or body/finger
correction was requested. This fixes attachment, not model contact generation.

Body discontinuity still drives a59.24cm sword/44.23cm shield step at frame9→10.
At half-frame samples, the baked independent world tracks separate from the
interpolated hand by up to8.39cm/6.35cm across that bad seam. Beyond it, midpoint
gaps peak at .228mm/.178mm. Marker equality at30Hz keys does not promise exact
continuous attachment; raw body continuity still needs a learned-motion fix.
Finger pose/handle closure, mesh collision and attack quality remain unresolved.
The intended held markers are satisfied; that is not a whole-animation quality
certificate. Native third-person and arms-only first-person candidate GIFs use
the same renderer/camera settings as the raw comparison, with private own geometry.
Advisory reports retain both variants and keep untested properties unknown.

Native advisory `review` HTML and its adjacent agent JSON are retained with all
three original clips and recipes. Local quaternion measurements are available;
custom-rig canonical body/ground, skin collision, balance and aesthetic checks
remain explicitly unevaluated. Hints did not block export, rank the trials by
count or trigger correction. Contact conclusions above use the explicit marker
contract and inspected action, not hint counts.
