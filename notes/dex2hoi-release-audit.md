# Dex2HOI release inventory and native host probe

Follow-up: the owner subsequently authorized manual weights and private tests.
The [native integration](dex2hoi-native.md) now downloads nothing at runtime and
has privately validated real HUMOTO inference. This page records the earlier
metadata-only investigation; its statements about work not yet done are historical.

2026-10-28 follow-up to [the feasibility audit](dex2hoi-feasibility.md).
Source remains pinned to
[`a92aca6b48b7a0c9a45a1465aa984213d3c8ceda`](https://github.com/cpratikaki/Dex2HOI/tree/a92aca6b48b7a0c9a45a1465aa984213d3c8ceda).
This work inspected public release metadata and executed an independently written
native host-math probe. It did **not** download checkpoint/statistic payloads,
run a learned network, establish native model parity or measure weapon quality.

## Actual public artifact inventory

The previous HTML fetch had succeeded, but its listing was inside a JavaScript
string assigned to `window['_DRIVE_ivd']`. Decoding only that string as JSON
reveals the release folders. No page JavaScript was executed. This resolves the
earlier inability to enumerate the release; the checkpoint files are publicly
listed, rather than merely advertised by the README.

The [linked release root](https://drive.google.com/drive/folders/1aCd23aVG9lh95DXYOIFfM3k2V_8uLAYO)
contains a [checkpoints folder](https://drive.google.com/drive/folders/1QncoRWKBp5oigIl3X2G0tCRX1rQ8kP9o),
with `grab`, `grab_eval`, `humoto` and `humoto_eval`. Only the two generation
folders were enumerated further. Evaluation contents were not downloaded.

| Variant | File | Public file ID | Reported bytes |
| --- | --- | --- | ---: |
| GRAB | eps.pt | `1pRywFP7AGAvVU31fuaDCP4kgE1ZTrQSB` | 1,237 |
| GRAB | mean.pt | `12YWAMMU7TXlx95QcRRcUmgjEqpAFa5jU` | 2,836 |
| GRAB | model.pt | `1il4-g1BmCg1-tNHtRJYvyUVYEpmk75JE` | 239,824,367 |
| GRAB | opt.pt | `1u3_QNdHSV-we-S7gh74Afb55jzqDWLS5` | 219,310,611 |
| GRAB | std.pt | `1qHTH14m5iU9H-qXemTWwA04sWzB6rgSd` | 2,829 |
| HUMOTO | eps.pt | `1l9KfxGvl8alupIlnMAZxysw2BsrEmA-g` | 848 |
| HUMOTO | mean.pt | `1faCYqZp66fq3t9p0KFSxSsxMmxjupHKh` | 2,829 |
| HUMOTO | model.pt | `1D0GKQzokllejYxSJPnRqlKlEZ56Edejy` | 723,196,239 |
| HUMOTO | opt.pt | `1jGQcrY4iW5Ox1fhT3TjXQ_UL4dCabPyZ` | 436,699,623 |
| HUMOTO | std.pt | `1w9UBUrnkq_TvB8p2mxnTyVFRAN6D3Q8c` | 2,824 |

Primary folder metadata:
[GRAB](https://drive.google.com/drive/folders/1fri-9oyVtelO6r0AhdgexispC1lW-u-u),
[HUMOTO](https://drive.google.com/drive/folders/1kCDC5_fInpo2WyV73oNrmiK-5R-pAWzW).
These are server-reported lengths, not independently hashed payloads. Model plus
normalization totals are 239,831,269 bytes for GRAB and 723,202,740 for HUMOTO,
excluding optimizer states. Runtime inference does not need those optimizer files.
Drive IDs and size are not immutable content pins: obtain full artifact hashes
before any native bundle is prepared or distributed.

Neither generation folder lists `args.json`, LICENSE, NOTICE or COPYING.
The pinned HUMOTO runner's `_load_train_args` **warns and returns an empty object**
when `args.json` is absent; it does not refuse automatically. Thus missing training
configuration is a compatibility/verification gap, not proof the upstream demo
cannot run. Constructor defaults are not verified checkpoint hyperparameters.
The reported statistic file sizes do not establish feature counts or normalization
values without inspecting their payloads.

Metadata snapshots, decoded listings and file identities are retained in ignored
`.research/dex2hoi/release-metadata.json`, with SHA-256 hashes of each metadata
snapshot. Snapshot hashes certify the audit input, not checkpoint content.

## What terms were actually found

The pinned Git tree contains only these license-named files:
`model/diffusion_planner/LICENSE` and
`model/diffusion_planner/utils/PYTORCH3D_LICENSE`. The public
[GitHub repository metadata](https://api.github.com/repos/cpratikaki/Dex2HOI)
reports `license: null`. A metadata-only fetch of the public
[HUMOTO model viewer](https://drive.google.com/file/d/1D0GKQzokllejYxSJPnRqlKlEZ56Edejy/view)
showed the same filename/size and no model grant or description containing one.
No top-level or checkpoint-specific grant was established. The paper's own license
is not automatically a license for code, weights or outputs, and licenses in
vendored subdirectories do not establish a grant for all new Dex2HOI code.

This is an unresolved release fact, not a conclusion that all local use is
forbidden. No grant has been invented and no model was added to Tide. The existing
developer checkout from the previous audit includes bundled `human_model/*.fbx`
and body-model JSON files; they were not used or copied into this probe, runtime,
Git changes or any release. Use an owner-authored explicit rig for native tests.

## Executed native proof: object representation host math

An independently written C++20 probe implements the 18D object's numerically stable
softmax over left/right/free position candidates, wrist-frame offset transformation,
and row-basis 6D rotation reconstruction. The representation was checked against
the pinned `obj_slot_utils.py` and `body_model_utils/geometry.py` definitions.
No upstream implementation was copied. This is necessary decode/GLB-adaptation
math, **not the network** and not a replacement motion generator.

The reference is independently evaluated in double precision with Python's standard
library, using 256 synthetic cases and seed 20261028. Cases include arbitrary wrist
poses, object offsets and logits of +1000/-1000 exercising every dominant branch.
The actual probe arithmetic uses native float32, is finite, and produces:

| Measurement | Observed result |
| --- | ---: |
| Cases | 256 |
| Maximum translation component error | 6.90842742124e-7 m |
| Mean translation component error | 8.8048316439e-8 m |
| Maximum rotation matrix component error | 9.16241631344e-7 |

Built and executed on Windows x64 with ClangCL 18.1.8 Release and warnings as
errors. No GPU, engine process, new dependency, model weight, body mesh or dataset
motion was needed. Python only prepared developer reference fixtures; the tested
executable uses the C++ standard library and runs independently.

Private reproduction files are under `.research/dex2hoi/native-probe/`:
`main.cpp`, `cases.json`, `cases.hpp`, `CMakeLists.txt`, configure/build logs and
`result.json`. The build is `cmake -S .research/dex2hoi/native-probe -B
.research/dex2hoi/native-probe/build -G "Visual Studio 17 2022" -T ClangCL -A x64`,
then `cmake --build .research/dex2hoi/native-probe/build --config Release` and
`.research/dex2hoi/native-probe/build/Release/dex2hoi-host-probe.exe`.

Input/probe pins:

- `main.cpp`: `06abfe425c41ece2cbcd6d1e82d8c380dac6f963ee2009fa5ed95ad1e26ece3d`
- `cases.json`: `081bac186e1bfe38df8a35f6014fdbd96fbbddc4a78fe052fd08382db6d0a595`
- `cases.hpp`: `168a39dfa5e42e9ade8466f4d17b1b3d6e9c01ce56c4405da778fbb39dc08731`

No PyTorch/native network comparison, body FK parity, normalization parity,
multi-window integration, contact quality or speedup follows from these numbers.

## Next concrete model implementation work

1. Establish actual artifact terms and inspect model/config/statistics separately
   from optimizers/evaluators. Pin content hashes, recover exact feature dimensions,
   object slot type, architecture, selected EMA/non-EMA state and normalization.
2. Build a dataset-independent input constructor from own explicit Mixamo-compatible
   history and licensed geometry: 10-frame prefix, initial transforms, masks, normal
   features and point samples must match the checkpoint's actual representation.
3. Separate CLIP, object encoder and motion graph exports. Precompute geometry
   features once per object. PointNet's farthest-point/radius/k-NN operations need
   faithful native host implementations or validated graph operators; the optional
   BPS branch cannot replace a differently trained PointNet checkpoint.
4. Implement native DDIM/CFG, exact prefix/chunk state, normalization and body/object
   decoding. The released forward inference loop uses `enable_grad=False` when the
   sampler supports it; it does not require HOIDiNi-style noise optimization.
5. Run fixed-input learned-network parity before complete generation, then validate
   independent object tracks and grip/event behavior using the explicit Tide tool
   contracts. A new backend must pass that tool-level evidence, not bypass it.

The release can now be located precisely and one native host operation is executed.
Learned inference and artifact clearance are still outstanding; no backend/default,
installer or model arithmetic in Tide changed in this investigation.
