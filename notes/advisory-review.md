# Local advisory motion review

`tide-motion review MANIFEST.json -o review.html` creates a self-contained offline
review page and `review.html.json` for agents. No new installed application,
renderer, inference model, network, server or dependency is required. Opening the
HTML in an existing browser enables two-view skeleton playback and human feedback.
Tide's pose editor remains the place for authoring guides. Source assets are never
changed. This is an initial calibration environment, not an automatic quality gate.

## Input

```json
{
  "version": 1,
  "title": "My animation review",
  "cases": [
    {
      "id": "walk",
      "label": "Case 01",
      "file": "walk.glb",
      "context": "Existing generated clip; no modifications.",
      "settings": {"groundY": 0, "rotationStepDegrees": 30}
    }
  ]
}
```

Paths resolve relative to the manifest. Cases accept optional `clip` (animation
name) and explicit `map` (existing humanoid mapping). A mapping retargets to SOMA
before measurement/display; this is a canonical proxy review, not a certification
of the original skinned mesh. Retarget root scaling and discarded translations
can change the motion. The original file, mapping hashes and presentation caveat
are retained. SOMA-77 is reduced to 30 for this body review; fingers are not checked.
Custom rigs without mappings receive local quaternion diagnostics and skeleton
playback, with canonical body/ground checks marked unavailable.
Display position tracks are rounded to .01 mm for compact pages; measurements use
the original float transforms. Display rounding does not modify exported assets.

Settings are sensitivity parameters, not correctness requirements:

| Setting | Default | Meaning |
| --- | --- | --- |
| `groundY` | 0 m | Explicit reference plane; no automatic ground estimation |
| `rotationStepDegrees` | 30 degrees/sample | Shortest parent-local quaternion step |
| `belowGroundMetres` | .02 m | Foot/toe joint-centre depth below plane |
| `nearGroundMetres` | .06 m | Height used by low-foot movement proxy |
| `lowFootSpeed` | .15 m/s | Horizontal speed of consecutive low foot/toe samples |
| `overlapMetres` | .02 m | Capsule overlap reporting threshold |

`props` optionally contains explicit bone-local capsule proxies:

```json
{"name":"test-blade", "bone":"RightHand",
 "start":[0,0,0], "end":[0,0.8,0], "radius":0.02}
```

These are authored test volumes, not meshes, inferred grips, weapon presets or
automatically calibrated sockets. SOMA torso radius is .13 m from Hips to Chest;
forearms have .035 m radius. Measurements use the nearest points of closed line
segments and summed radii. Only forearm/torso and supplied prop/torso pairs are
tested. Enabling analysis capsules in the viewer exposes these assumptions.

## Observation contract

Findings contain rule ID, subjects, inclusive sampled frame interval, seconds,
peak frame/value, units, sensitivity threshold, measurement method and an initially
unreviewed interpretation. Adjacent threshold exceedances merge; gaps split.
Changing local quaternion sign alone is not motion. Rotation steps are reported
per sample, with FPS retained. Low-foot movement is not detected planted contact.
Foot centres are not soles. Capsule overlap is not skin penetration. No observations
does not mean good motion. No severity, pass/fail or aesthetic score is assigned.

Coverage explicitly marks anatomical reach/joint limits, physical balance, mesh
collision, swept/continuous collision and aesthetics as not evaluated. Unknown
must not become a zero or a pass. All checks are native offline analysis and never
replace learned generation, correct poses, reject unusual art or alter the GLB.

The agent JSON also retains `measurements` for every evaluated rule/subject,
including those with no hints: eligible sample count, exceedance count, mean,
maximum and peak frame. Rotation statistics use adjacent pairs (exclude frame 0).
Low-foot motion statistics use only pairs whose two joint centres meet the height
condition; zero eligible samples yields null statistics, not zero movement.
These are continuous measurements, not a quality score. Comparing hint counts
alone can select a worse candidate because intervals split/merge at thresholds.
Keep frame counts, authored poses, sensitivity settings and sample eligibility
matched, and inspect action adherence before selecting a candidate.

Bounds: 1 MiB manifest, 1..24 cases, 128 joints/case, existing 18000-frame clip
bound, 60000 total frames, 16 prop proxies/case, 10000 observation intervals/case,
64 MiB HTML. All inputs are checked before either output is written. Existing
outputs/partials are refused. Reports contain local paths and source motion;
keep review bundles containing purchased assets private.

## Human calibration

Watch before revealing hints. The source/context disclosure can reveal test
treatments; it stays collapsed initially. Assess a finding as accurate and needing
attention, accurate but acceptable, intentional, inaccurate or unsure. Add reasons
and timestamped notes, including issues with no matching finding. Notes retain
nearby observation IDs (within .5 seconds of the peak or inside its interval) as
possible associations, not a claim that the tool explains the user's concern.

Feedback stays in the open view. Export JSON before closing; there is no promised
browser persistence or silent upload. Import resumes feedback only for the exact
source/mapping identities, selected animation, settings and rules version.
Import validates fields/timestamps and recomputes associations. Text is rendered
as text, not HTML. Embedded JSON escapes `<` to prevent script termination.

Feedback is separate from immutable measurements. Intentional exceptions preserve
the observation. Nothing auto-repairs or learns from one rating. Cross-version
matching, motion revisions, aggregate calibration statistics and direct Tide pose
editor feedback integration remain future work. Human ratings are the evidence
needed to decide which diagnostics should change next.

## Reproduce the owner's local calibration set

Build the optional, uninstalled fixture helper, then run the setup script with
the owner's existing licensed assets and installed MotionBricks weights:

```powershell
cmake --build build --config Release --target review-examples
./tools/create-review-set.ps1 -Tool ./build/Release/tide-motion.exe `
  -FixtureTool ./build/Release/review-examples.exe `
  -AssetsRoot D:/Projects/tide-assets -Output ./output/my-review
```

Optionally add `-ExistingGeneratedClip PATH.glb` for a ninth case. The script
uses the library's Polygon idle, forward parry and forward dodge roll, explicit
Synty mapping, native pose extraction and actual MotionBricks inference between
guides. It also makes labeled diagnostic edits of an idle and a deliberately
misaligned prop capsule. Sources are never modified or copied into Git. The
script refuses an existing output directory to preserve any prior human work.
