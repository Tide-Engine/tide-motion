# MoMask controls and model limits

MoMask is an alternative masked-token model, rather than an interchangeable
checkpoint for Kimodo or ARDY. Its pinned transformer interfaces accept CLIP
text, whole-body motion tokens, quantizer level/history and padding. They have
no spatial path, effector target or Kimodo constraint-CFG input. Primary source:
[the pinned transformers](https://github.com/EricGuo5513/momask-codes/blob/94a6636c9c463b7a9414c3401a6f1b67e6c51824/models/mask_transformer/transformer.py).

| Feature | MoMask in Tide Motion |
|---|---|
| Text generation, seed, duration, guidance | Yes; ASCII prompts, .2..9.8 s, native 20 Hz converted to SOMA at 30 Hz |
| Temporal inpainting | Yes; whole-body tokens covering [begin,end), at least 12 canonical frames; maximum source 293 frames |
| Outside poses and edit endpoints | Preserved, including all original SOMA-77 extra-joint tracks; endpoint body/root correction is explicit postprocessing |
| Batch and session reuse | Yes; earlier outputs may feed later actions; sessions reload when cache/device/threads change |
| Variants and contact sheet | Yes; --variants 1..64, sequential seeds, native session reuse, same naming/renderer as the other backends |
| CPU thread/device controls | --threads 1..1024 (default 4), --device cpu (default), auto or gpu |
| GPU | Native DirectML acceleration on Windows x64, with CPU auxiliary nodes and CPU CLIP text; gpu is strict, auto falls back |
| In-place | --in-place on generation; the existing native root-travel removal, recorded as postprocessing |
| SOMA-77 | --rig soma77 on generation adds static relaxed hands; inpaint preserves a SOMA-77 source and its extra tracks, with no new predicted fingers |
| Retarget, preview, mirror, cleanup, MotionBricks connect | Existing commands consume these ordinary GLBs; mappings/contact plans/other model assets retain their normal requirements |
| Kimodo start/end/keyframe/effector/path flags | No; temporal source context is not the same interface as spatial constraint conditioning |
| ARDY history/long-form/changing-prompt controls | No; the tested MoMask window remains bounded; no pose-blended long-form substitute |
| --model v1.1/v1.2 | Kimodo checkpoint selection only; MoMask uses its own exact pinned manually supplied bundle |

Examples (weights already imported):

```sh
tide-motion generate "A person walks forward." --backend momask --duration 3 --seed 7 --variants 4 --threads 4 -o walks.glb
tide-motion generate "A person walks forward." --backend momask --device gpu --rig soma77 --in-place --duration 3 -o walk77.glb
tide-motion inpaint "A person waves with the right hand." --source walk77.glb --begin 18 --end 66 --device gpu -o wave77.glb
```

Variants save `walks.seed7.glb` through `walks.seed10.glb` and attempt a contact
sheet. `--tide PATH` selects the renderer; `--variants-gif` additionally uses the
existing bounded animation renderer, requiring Tide 0.5.0-nightly.45+ with
`run --record`. Rendering failure keeps completed assets and reports the reason.
Generation works without Tide. The locally validated engine at 28b61e4db supports
the PNG capture but lacks `--record`, so a GIF was refused, not claimed tested.
`--in-place` is refused for inpaint because changing the whole root track would
violate outside-range preservation. An explicit `--rig soma30` is refused for
a SOMA-77 source for the same preservation reason; inpaint also refuses changing
a SOMA-30 source into SOMA-77. Select the presentation rig during generation first.
Custom rigs still require
explicit existing retargeting; there is no automatic rig/bone identification.

GPU inference shares the already pinned/packaged Microsoft ORT 1.24.4 DirectML
runtime and DirectML 1.15.4 used by ARDY; CPU ORT stays 1.31.1. Numeric arrays alone
cross runtime boundaries. Notices/redistribution terms and exact URLs/hashes are
in [ardy-native.md](ardy-native.md). No new model export or asset pin is required.
`MOTION_MOMASK_DIRECTML` controls packaging, initially following the ARDY setting
on fresh builds; it can be selected independently. CPU remains available. GPU
loss in auto restarts the entire current job from its seed on CPU, before writing
an asset; later jobs reuse the fallback. Runtime-loss injection was not performed;
missing-runtime strict refusal and actual auto CPU inference were tested.

## Executed checks

Windows x64 ClangCL Release, RTX 4080 SUPER, TIDE_HIDDEN=1, isolated TEMP/TMP;
all GPU inference/previews serialize under TideMotionGpuValidation. All twelve integrated
model-free CTests pass. WSL GCC warnings-as-errors syntax checks also pass.
No evaluation dataset or research body mesh/assets are used.

`momask-gpu-tests PRIVATE_EXPORT_DIRECTORY`: all twelve real PyTorch-reference
motion-network cases pass at 4/7/49 tokens. GPU float tolerance is fixed at
.002 + .002*abs(reference), independently of the unchanged tighter CPU bound.
Measured max absolute errors: masked 9.78e-6, residual 2.39e-5, RVQ decoder
.00155258; all RVQ encoder indices match exactly. CLIP remains the previously
validated CPU tower. Native provider profiles show DML kernel events in all four
motion graphs (699 masked, 705 residual, 135 decoder, 456 encoder across the
three cases), with CPU auxiliary operations too. This is not all-operator GPU
execution. Sampled per-process GPU local peak: 351,043,584 bytes; not total VRAM
or a guaranteed transient peak. No CUDA/Python/user SDK is required.

A seven-action real native batch exercises CPU generation, cold/warm GPU repeat,
SOMA-77 idle/inpaint, in-place variants with a captured contact sheet, and a CPU
thread change. All eight outputs have 89 frames, finite local/FK transforms and
validated SOMA rigs. Walk CPU/GPU base and residual tokens match; warm GPU tracks
are identical. Independent official-tokenizer/full-sampler reference matches.
These observations do not promise universal cross-device token/bitwise equality.

Same prompt/seed, 3 s requested, 18 mask steps: CPU generation .1364 s, first GPU
.4916 s, warm GPU .1469 s, excluding session construction but including native
sampling/conversion. A single local observation, not a general benchmark: GPU
did not win on these short clips, so CPU remains MoMask's compatibility default.
The source and GPU math/sampler/model pins are unchanged. Previous MoMask versus
Kimodo contact-quality limitations remain; this control work does not improve
the model's foot contacts or establish general quality equivalence.

`tests/momask-controls.py` independently checks exported tracks/FK and authors one
varying finger track on a self-generated source to detect accidental loss. Real
native GPU editing preserves every extra-joint frame and all outside/end poses,
including boundary root/FK continuity; largest local-component error after
quaternion sign matching is 5.97e-8 on that repeat. In-place keeps rotations and
removes net horizontal displacement in the measured example. The contact sheet
shows plausible walking poses; sparse images do not establish all-frame quality.

## Controls that need more model/sampler work

Adding flag names alone cannot supply spatial learned conditioning. Full pose
anchors might be investigated as RVQ token context, and path/effector control
might use separately validated constrained token search or a compatible trained
checkpoint. These are unimplemented technical research tasks, not a conclusion
that the licensing issue makes them technically impossible. Exact IK/path warping
would be explicit postprocessing and would not demonstrate that MoMask learned
the requested constraint. Existing cleanup remains a separate operation.

Likewise, extending past 196 native samples would require a tested windowed
token-continuation sampler with seam, drift and prompt-change quality checks;
the current checkpoint does not expose ARDY's history-conditioned interface.
Neither capability is claimed here. Checkpoint commercial rights remain a separate
unresolved question; the owner-local weights are private and manually supplied.
