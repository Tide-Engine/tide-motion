# Native equipment authoring

Equipment plans add authored object motion, measured grip intervals and optional
guarded arm correction to an existing body animation. They run offline in the
native executable, without model loading, Python or a service. Body generation
remains the learned backend's job. These controls do not make that backend
object-aware and do not run in Tide deterministic match simulation.

## Commands and review loop

```sh
tide-motion equipment inspect BODY.glb --plan EQUIPMENT.json --report NEW-report.json
tide-motion equipment inspect BODY.glb --plan EQUIPMENT.json --correct-grips
tide-motion equipment bake BODY.glb --plan EQUIPMENT.json --correct-grips --strict -o NEW.glb
```

`inspect` prints the effective report as JSON. `--report NEW.json` also saves it;
it is available for all three actions. `--correct-grips` evaluates a corrected
body in memory for inspection, baking or preview. It does not overwrite the input.
`bake` writes the combined GLB and `NEW.glb.json` recipe/report. Optional
`--body-output BODY-ONLY.glb` also saves the corrected body without equipment
nodes, plus its recipe, for ordinary body preview, retargeting and further editing. Output, sidecar and
partial-file collisions are refused. Use distinct output paths for each attempt.
`--clip NAME` selects an animation when the source contains several.

`--strict` refuses a bake or preview if any final sampled grip misses its position
or enabled orientation tolerance. It can also be used with inspect, where failure
occurs before printing the report; inspect without `--strict` to see residuals.
A run with no grip intervals has no grip failures to check. Strict mode does not
certify collision freedom, timing, anatomical plausibility or action quality.

Review the source body plus the plan directly:

```powershell
$env:TIDE_HIDDEN='1'
tide-motion equipment preview BODY.glb --plan EQUIPMENT.json --correct-grips `
  --capture equipment.png --frames 31 --tide C:/path/to/validated/tide.exe
tide-motion equipment preview BODY.glb --plan EQUIPMENT.json `
  --setup examples/equipment/camera-first-person.json `
  --capture equipment-fps.png --frames 31 --tide C:/path/to/validated/tide.exe
```

Use a Tide executable built from the engine checkout being validated. Preview
runs must be bounded with `--frames` or `--capture`. `--gif NEW.gif` is supported;
an explicit frame bound is useful for repeated comparisons. `--first-person`
uses the default eye camera. `--setup CAMERA.json` loads the existing version-1
preview camera configuration and cannot be combined with `--first-person`.
Equipment setups must have no static `props`: dynamic props belong in `--plan`.
The supplied camera example is camera-only and explicitly uses SOMA `Head`.

On a custom character, use `--character MODEL.glb|.fbx --map BONES.json` for a SOMA
source; the plan's bone names must refer to the resulting character rig. An
already-custom source uses the matching character and omits `--map`.
`--texture IMAGE.png` is a preview option. Supply a skinned arms mesh for custom
first-person review; the built-in first-person mannequin draws arms only while
retaining the complete SOMA hierarchy. No automatic bone identification, mesh
cutting or skinning is performed. Keep required root/spine ancestors in arms rigs.

## Version-1 plan structure

Top-level fields are `version` (required, `1`), `objects` (required), and optional
`grips`, `strings`, `events`, `protectedFrames`. Unknown fields are refused. Names
are nonempty strings of at most 128 bytes. Object names must be unique and distinct
from body bone names. Paths in `mesh` resolve against the plan file's directory;
the effective report records resolved paths and hashes.

Frame indices are absolute, zero-based integers in `0..17999`, not relative to a
segment or contact interval. Intervals include **both** endpoints. Every object's
segments must start at frame 0, meet without gaps or overlaps, and finish at the
source clip's last frame. A five-frame source therefore uses frames 0 through 4.
The source loader samples its selected animation on Tide Motion's body timeline;
use the report's `frames` and `fps`, rather than assuming an imported file's labels.

### Objects and anchors

Each of 1–32 objects contains `name`, optional `mesh`, and 1–256 `segments`:

```json
{
  "name": "sword",
  "mesh": "meshes/sword.glb",
  "segments": [{
    "start": 0,
    "end": 4,
    "space": "bone",
    "anchor": "RightHand",
    "interpolation": "linear",
    "keys": [
      {"frame": 0, "position": [0, 0, 0], "rotation": [0, 0, 0], "meshScale": [1, 1, 1]},
      {"frame": 4, "position": [0, 0, 0], "rotation": [0, 0, 0], "meshScale": [1, 1, 1]}
    ]
  }]
}
```

`space` is `world` (default), `bone` or `object`. World space omits `anchor`.
Bone space explicitly names a body bone; object space names another declared
object. Cyclic object anchors are refused. A mesh-less object is a useful named
transform, such as a nock or grip guide, and is still exported as an animated node.

Each key requires `frame`; optional `position`, `rotation`, `meshScale` default to
zero, zero and one respectively. Key frames must be strictly increasing and span
the segment's endpoints exactly. Positions are metres, rotation is Euler degrees
with Tide's Z-then-X-then-Y convention. Components must be finite; positions are
within ±10,000 m, rotations within ±3,600 degrees, mesh scales are positive and
within `0.000001..1000` per axis.

`linear` interpolation uses linear positions/scales and quaternion interpolation
after converting Euler keys. `step` holds the preceding authored key **at sampled
frames**. Export bakes a translation/rotation/scale sample at every body frame with
GLB LINEAR interpolation; it does not preserve subframe step discontinuities.

Anchors are rigid world frames with their inherited scale removed. A centimetre
rig still places the socket correctly in metres, but does not shrink a held mesh
or its metre offsets. `meshScale` affects geometry only. It does not change another
object's anchor frame, a grip target offset, or string endpoint/nock offsets.
All resolved objects export as independent world-space root tracks; object-space
authoring is baked, not retained as an inheriting scaled object hierarchy.

At a segment boundary, author the new frame in the new coordinate space. The tool
does not infer a seamless handoff or velocity match. `anchorTransitions` reports
position and rotation steps at these boundaries, separately from grip acceptance.

### Grips and guarded correction

Each of at most 64 grip intervals requires `name`, `object`, `bone`, `start`, `end`.
`position` and `rotation` define a target in the object's unscaled local frame.
Optional `handOffset` has `position`/`rotation` in the body's named bone frame,
allowing a palm/socket offset from the wrist. The measured equality is:

`worldHandBone × handOffset = worldObject × gripTarget`.

| Field | Default | Accepted range / meaning |
| --- | --- | --- |
| `orientation` | `true` | Include rotation in acceptance; `false` checks position only. |
| `positionTolerance` | `0.005` | `0..0.5` metres, maximum sampled residual. |
| `rotationTolerance` | `5` | `0..180` degrees, maximum sampled residual when orientation is enabled. |
| `maxCorrection` | `0.25` | `0..1` metres, bounds requested wrist correction distance. |
| `maxRotation` | `90` | `0..180` degrees, bounds change in a chain joint's local rotation. |
| `maxStepIncrease` | `30` | `0..180` degrees, bounds added local rotation step, including interval boundaries. |
| `chain` | omitted | Required for correction: exactly `[upperArm, forearm, hand]`, ending at `bone`. |

The three correction bones must be direct parent/child links below the motion
root. A chain containing intervening twist/helper bones is not accepted as a
direct chain. Overlapping correction intervals cannot share chain joints.
Diagnosis does not require `chain` and can measure another explicitly named bone.

Without correction, all body transforms are preserved. With correction, only the
explicit chain's local rotations can change inside the interval. Local body
translations, root transforms, local finger animation, and protected frames stay
unchanged. Descendants still move with corrected ancestors; this is not a claim
that finger world trajectories are fixed. No finger contact or collision solver
is added.

Object trajectories are resolved from the **original** body first and held fixed
during IK. Correcting a hand does not silently move its target object, and a
successful correction cannot make its own measurement pass by dragging the object.
Conversely, a primary object rigidly following a hand can match that primary grip
by construction; its zero residual alone does not establish a good weapon pose.

The solver rejects unreachable/degenerate targets, excessive position/rotation
correction, excessive added joint steps, and unresolved/protected-frame residuals.
A rejected grip interval rolls back its attempted body edit in full. Earlier
accepted disjoint corrections remain applied. Final reports remeasure every grip
on the resulting body, so rejection status and overall satisfaction are distinct.

`protectedFrames` explicitly lists immutable frames. Source recipe
`BODY.glb.json` protected frames are included automatically. Protect authored
endpoints/keyframes when appropriate; a protected frame can cause an interval to
be rejected if its grip is not already within tolerance.

### Bow articulation, strings and release

Articulate a bow by authoring separate rigid riser/upper-limb/lower-limb objects.
This does not deform a single skinned bow mesh. Put each limb mesh's origin at
the intended hinge, then author its object-local rotations/positions.

Each of at most 16 `strings` requires `name`, `object`, `top`, `bottom`, `nock`.
The named `object` must exist. `top` and `bottom` are local metre points. Optional
`topAnchor`, `bottomAnchor`, `nockAnchor` each name an existing authored object;
they default to `object`. These are **object frame names**, not body bones.
Use a bone-anchored helper object if an endpoint must follow a body bone.

`nock` is an ordered list of `{ "frame": N, "position": [x,y,z] }` point keys,
spanning frame 0 through the last body frame. Its positions interpolate linearly
in `nockAnchor`'s unscaled frame. `thickness` defaults to `0.004` metres and must be
`0.0001..0.1`. The tool emits two animated rigid pieces named `NAME.top` and
`NAME.bottom`, each a centred unit box along +Y scaled to endpoint-to-nock length.
These part names must not collide with body/object names. Zero-length pieces are
refused. Strings are authored render geometry, not a tension/elastic simulation.

Each of at most 256 `events` requires `name` and `frame`, with optional `object`
and arbitrary JSON `data`. The tool adds `timeSeconds = frame / fps`. The exact
name **`release`** requires an authored object and validates that its event frame
is the start of a new `world` segment immediately after a `bone`/`object` segment
for that object. The authored segment transition performs detachment; the event
validates and records it. Other names are annotations.

Arrow flight is explicitly keyed in world space. Release does not generate a
projectile trajectory, gravity, collision, impact, disappearance or an animation
trigger in the consuming game. Match release position/orientation and velocity
yourself, inspect the transition report, and consume the event in your game if
needed. The bow example includes a separate mesh-less nock frame so the string
and nocked arrow use the same authored draw point.

## Reports and exported assets

Each grip report includes its `status` (`diagnosed`, `accepted`, `rejected`),
rejection `reason`, attempted maximum joint correction/step increase, and `before`
and final `after` measurements. Measurements include sampled count, mean/maximum
position and rotation residual, worst position frame, mean per-frame change of
object-relative error, and `satisfied`. The top-level report has
`allGripsSatisfied`, frame count/fps, correction choice, `anchorTransitions`,
`objectTrajectoryBasis`, `collisionChecked: false`, and the effective plan.
CLI reports add source/mesh hashes, paths, protected frames and optional character/
mapping identities. Event time is available in exported events/recipe.

The GLB contains the body **skeleton animation**, independent equipment roots,
equipment geometry and sampled tracks. It does not embed a skinned character/body
mesh. `asset.extras.tideMotion` stores the equipment version, body motion-root
name, fps/frames, events, report, mesh-conversion records and
`bodyMeshEmbedded: false`. These are ordinary GLB nodes, accessors and animations;
the extra metadata does not make events execute automatically in another engine.

For appearance-preserving equipment export, supply a static rigid self-contained
GLB with one embedded binary buffer. The exporter retains its static scene
hierarchy, vertex attributes, core materials, embedded PNG/JPEG textures and
samplers; `KHR_materials_unlit` and `KHR_texture_transform` are supported. Animated,
skinned, morph-target, camera-containing or externally buffered/textured GLBs are
refused, as are unsupported extensions. Separate articulated pieces belong in the
equipment plan. Source asset metadata is retained under the object root's extras.

Static rigid FBX equipment is converted through pinned ufbx in Y-up metres, with
triangulated positions/normals and base color/opacity/metalness/roughness conversion.
Animated, skinned, morphing and textured FBX equipment is refused. Other FBX shader
features, UVs and vertex colors are not represented; the conversion record states
that limitation. Use self-contained GLB when those appearance features matter.
No third-party mesh or model is bundled by these examples.

The body-only animation loader is not an equipment-scene importer. Combined GLBs
contain independent object roots and potentially nonuniform/animated equipment
scale, which that loader refuses. To review another iteration, use
`equipment preview SOURCE-BODY.glb --plan EQUIPMENT.json`; consume the combined GLB
in the engine/game or an ordinary glTF-capable tool. Do not feed the baked equipment
asset back as the source body for another bake.

## Five-frame authoring templates

- [sword-shield.json](../examples/equipment/sword-shield.json): one persistent
  body-bone attachment and primary grip per hand.
- [two-hand-grip.json](../examples/equipment/two-hand-grip.json): an independent
  world object path, explicit left/right three-bone correction chains over frames
  1–3, with frames 0 and 4 protected.
- [bow-release.json](../examples/equipment/bow-release.json): rigid articulated bow
  limbs, a mesh-less shared nock frame, two string pieces and arrow release into
  an authored world track at frame 3.
- [camera-first-person.json](../examples/equipment/camera-first-person.json): a
  camera-only setup for equipment preview, with explicit SOMA `Head`.

All motion plans cover exactly **five body frames**. Replace placeholder paths
under `examples/equipment/meshes/` with your licensed assets, match their origins,
and calibrate every position/rotation/grip to the actual source rig and action.
The two-hand example intentionally uses broad demonstration rotation/step bounds;
choose bounds appropriate to your authored clip before accepting production edits.
The bow's world arrow keys are illustrative coordinates, not a computed release
from an arbitrary source pose. Extend keys/intervals explicitly for longer clips.
These are schema/authoring templates, not validated model-generated combat motion
or weapon-quality presets.
