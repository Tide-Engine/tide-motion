# Prompting Tide Motion

What to write in a `generate` prompt, and what to try when a result is wrong.
Every claim here was measured with Tide Motion 0.5.0 on an RTX 4080 SUPER: 8
seeds per prompt (`--variants 8`), 2.5 seconds, default settings, unless a
table says otherwise. Results were scored by forward kinematics on the output
GLBs: a **kick** is a foot 0.45 m above the floor, **airborne** is both feet
0.08 m up at once, a **crouch** is the head dropping 0.5 m.

## The short version

1. Describe **one movement**, plainly: `A person kicks forward with the right leg`.
2. Give it **1.5–3 seconds**. Chain more moves with `--then`.
3. Name the **side** and the **direction** you want; leave out props, clothes,
   characters and scenes.
4. Ask for the **change** you want, not the state: "a standing person crouches
   down", not "a person crouching".
5. Make `--variants 8` and pick from the contact sheet.
6. Still not there: reword, then raise `--guidance` to 3–4, then pin the moment
   with a `--keyframe` or `--effector` taken from your best variant.

## What to write

Kimodo learned from motion capture described in short plain sentences. It
understands bodies moving, not stories, objects or characters.

| Prompt | Result (8 seeds) |
|---|---|
| `A person kicks forward with the right leg` | 8 right kicks; steps forward 1.6 m |
| `Kick forward with the right leg` | 8 right kicks; same as above |
| `The person kicks forward with their right leg.` | 8 right kicks; same |
| `A man kicks forward with the right leg` | 8 right kicks; same |
| `A person kicks forward with the left leg` | 8 left kicks |
| `A person kicks` | 7 kicks, 4 right and 3 left |
| `kick` | 8 kicks, 7 of them left, walking 1.7 m |
| `A person does a high front kick with the right leg` | 6 right kicks (2 left), foot 1.08 m high, in place |
| `A person does a roundhouse kick with the right leg` | 8 right kicks, foot 1.34 m high, in place |
| `A person quickly kicks forward with the right leg` | 7 right kicks, a little lower; no faster |
| `A person holding a sword kicks forward with the right leg` | 3 kicks, foot only 0.39 m high |
| `A person kicks forward with the right leg and then kicks forward with the left leg` | no clip with both kicks |
| `--then`: right kick 1.5 s, then left kick 1.5 s | 8 clips with both kicks |

What this means:

- **Wording style doesn't matter.** "A person", "A man", "The person … their",
  an imperative or a full stop all gave the same results. Keep "A person …".
- **Name the side.** Without it the model picks one, and bare `kick` mostly
  kicked left.
- **Direction words move the root.** "forward" made every kick step 1.6 m
  forward; "high front kick" and "roundhouse kick" stayed in place. Leave out
  "forward" (or add "in place") when the clip must not travel, or use
  `--in-place` to remove travel afterwards.
- **Named techniques work.** "roundhouse kick" and "high front kick" came out
  higher and cleaner than "kicks forward" (all 8 roundhouses kicked, foot at
  1.34 m). Try the move's usual name before describing it limb by limb.
- **Size words work, speed words barely do.** "high" and "as high as they can"
  changed the movement; "quickly" didn't make it faster. Change the duration
  for speed instead.
- **Props weaken the action.** Mentioning a sword cut kicks from 8 to 3. The
  model has no sword, so describe only the body; hold a weapon stance with a
  start pose or effectors instead.
- **One action per prompt.** "kicks … and then kicks …" in one prompt failed
  every time; the same two prompts with `--then` worked every time.

### Moves versus states

A prompt describes the whole clip. Asking for a state gives you that state from
the first frame to the last:

| Prompt | Result (8 seeds) |
|---|---|
| `A person crouches down` | already crouched at frame 0, crouched throughout |
| `A person crouches down low to the ground and stays there` | the same |
| `A person standing crouches down` | 8 go from standing to crouching (head drops 0.9 m) |
| `A person goes from standing to crouching` | 7 of 8 |
| `A person crouches down` with `--start-pose` on a standing idle | 7 of 8 |
| `--then`: `A person stands still` 1 s, then `A person crouches down` 1.5 s | 5 of 8 |

For a transition, say where the movement starts ("a standing person …",
"from standing to …"), or better, start from your own pose with
`--start-pose`, which also makes the clip begin exactly where your game is.

### Jumps

| Prompt | Airborne | Height | Travel |
|---|---|---|---|
| `A person jumps` | 6 of 8 | 0.10 m | 1.0 m forward |
| `A person jumps as high as they can` | 8 of 8 | 0.39 m | in place |
| `A person jumps up in place with both feet` | 8 of 8 | 0.44 m | in place |

Plain "jumps" mostly hops forward; say how and where.

## Duration

One move fits in about 1.5–3 seconds. Given more time, the model does not
stretch the move; it fills the time with standing, and the move often vanishes:

| Duration | `A person kicks forward with the right leg` | long sword-stance prompt¹ |
|---|---|---|
| 1.5 s | 5/5 | 0/5 |
| 2.0 s | 5/5 | 5/5 |
| 2.5 s | 5/5 | 4/5 |
| 2.7 s | 5/5 | 2/5 |
| 3.0 s | 5/5 | 3/5 |
| 3.5 s | 3/5 | 1/5 |
| 5.0 s | 1/5 | 0/5 |

¹ `A person holding a sword in a fighting stance kicks forward with the right
leg, then returns to the stance`

The short prompt is more robust: in a later run on seeds 0–7 it kicked 8 of 8
at 4 seconds. Start and end poses make long clips harder, though: between idle
poses it kicked 18/24 at 2.5 s but 12/24 at 4 s. For a longer clip, chain
prompts with `--then` (each 1.5–3 s), or pad the move with a short
`A person stands still` segment.

## Nudging a result

Work down this list; each step costs more control or more realism than the one
before.

1. **More seeds.** Results vary a lot by seed: one prompt gave a clean kick, a
   rough kick and no kick on three seeds. `--variants 8` takes about 15 seconds
   on a recent GPU after loading and writes a contact sheet; any variant can be
   remade exactly with its seed.
2. **Reword.** Shorter, one movement, a named technique, the side, the
   direction, and the change rather than the state (see above).
3. **Shorten.** If the move is missing or half-hearted, try 2 seconds.
4. **Raise `--guidance`** (default 2) to 3–4 when the move still doesn't
   appear, especially in clips over 3 seconds. Higher guidance follows the
   prompt harder but stiffens and exaggerates the whole body (a boxer's guard at
   4, a caricature at 6) and slides the feet more. Guidance 1 barely moves.

   | `--guidance` | Kicks, 2.5 s | Kicks, 4 s | Foot slide (cm/frame), 2.5 s / 4 s | Pose error mean / max (cm) |
   |---|---|---|---|---|
   | 1 | 0/8 | 0/8 | 0.15 / 0.06 | 2.0 / 4.8 |
   | 2 (default) | 18/24 | 12/24 | 0.22 / 0.15 | 2.1 / 5.3 |
   | 3 | 15/24 | 21/24 | 0.24 / 0.21 | 2.1–2.4 / 5.8–6.3 |
   | 4 | 23/24 | 23/24 | 0.29 / 0.29 | 2.3–2.8 / 6.4–7.5 |
   | 6 | 8/8 | 8/8 | 0.39 / 0.45 | 2.4–4.1 / 6.8–10.0 |

   (Kicks between idle start and end poses; foot slide is the planted foot's
   mean horizontal movement; pose error is the model's distance from the start
   and end poses before they are made exact.)

5. **Pin the key moment.** Take the best frame from any variant, or from an
   existing clip, and require it: `--keyframe 40 best.glb@38` for the whole
   body, or `--effector right-foot 40 best.glb@38` for just the foot. Every seed
   then has to pass through it, and the prompt only shapes the motion around it.
6. **Fix it after generating.** `mirror` makes the other side (a right kick into
   a left kick); `--in-place` removes travel; `connect` bridges two clips.

### Starting and ending in your game's poses

`--start-pose` and `--end-pose` make the clip begin and end exactly in your
poses, so it cuts cleanly from and back to your idle. They cost little: kicks
between idle poses kicked 7 of 8 times against 8 of 8 without. Leave
`--constraint-guidance` at 2: lower lets the model stray from the poses (5.5 cm
mean before correction, against 2.1), higher suppresses the action, especially
in longer clips:

| `--constraint-guidance` | Kicks, 2.5 s | Kicks, 4 s | Pose error mean / max (cm), 2.5 s | 4 s |
|---|---|---|---|---|
| 1 | 8/8 | 5/8 | 5.5 / 12.9 | 3.7 / 9.2 |
| 2 (default) | 7/8 | 2/8 | 2.1 / 5.1 | 2.1 / 5.0 |
| 3 | 7/8 | 1/8 | 2.2 / 5.2 | 2.1 / 4.7 |
| 4 | 7/8 | 1/8 | 2.3 / 5.4 | 2.1 / 4.8 |
| 6 | 6/8 | 0/8 | 2.9 / 6.0 | 2.4 / 5.6 |

Keep `--correction-frames` at its default of 10. With 0, the constrained frame
alone is set: on a kick whose hands began 60 degrees from the start pose, that
frame popped 60 degrees (30 with 2 frames, 22 with 4).

A pose from a character that isn't SOMA goes through `retarget --to-soma`
first; check its preview against the original, as a converted pose is only as
faithful as the bone map.

## What doesn't help

- **Describing the character, props, clothing or scene.** It weakens the action
  (see the sword row above).
- **Negative prompts.** Steering away from "standing still" or "walking" was
  tried and dropped: "walking" removed kicks entirely, and only a full sentence
  ("A person stands still doing nothing") helped, about as much as guidance 3.
- **Long clips for one move.** See Duration.
- **More diffusion steps.** Kicks out of 16 (the long sword prompt at 2.5 s
  and the short prompt at 4 s, 8 seeds each): 10 steps 7, 25 steps 15, 50 steps
  15, 100 steps 13. Above 50 only costs time; 25 is fine for drafts at half the
  time; 10 failed every 4-second clip. A seed's result depends on the step
  count, so remake a variant with the same `--steps`.

### Travel, endpoint fit and counted steps

When both pose files occupy the same place, default placement asks the motion to
return there. Use `--end-at X,Z[,YAW]` to finish in the same body pose elsewhere,
or `--placement free` to let a learned guide choose its location. Coordinates
are absolute world metres. Inspect displacement, path length and peak speed on
the variant sheet; a large peak can reveal an endpoint jump.

The reported model rotation error precedes correction. Full-body correction
makes the selected pose exact, including sword-idle wrist/spine rotations;
`--no-correction` retains the measured residual. It does not make the intervening
motion or velocity exact. A wider correction window spreads that residual.

"Two steps" in text is a preference, not a count constraint: the model may take
2�4. There is no reliable exact step-count control in this backend. For explicit
footfall timing use authored SOMA targets with `pose-sequence` and inspect the
result; intermediate contacts remain a model quality limitation. Do not shorten
or time-warp a clip and assume that enforces the requested step count.
