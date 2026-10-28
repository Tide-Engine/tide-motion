"""Independent GLB measurements for the Kimodo capability runs; development only.

Reads Tide Motion GLBs with its own decoder and forward kinematics (standard library
only; no tool code). Foot skate follows the definition of upstream Kimodo's
FootSkateFromHeight (kimodo/metrics/foot_skate.py at e1c219c: mean toe speed while a
toe is below 5 cm), computed here independently. Not used by the tool or its CTests.

python tests/capability_metrics.py WORK_DIR > metrics.json
"""
import json
import math
from pathlib import Path
import struct
import sys

FPS = 30.0


def payload(path):
    raw = path.read_bytes()
    size = struct.unpack_from('<I', raw, 12)[0]
    return json.loads(raw[20:20 + size]), raw[28 + size:]


def mul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w * X + x * W + y * Z - z * Y, w * Y - x * Z + y * W + z * X, w * Z + x * Y - y * X + z * W, w * W - x * X - y * Y - z * Z)


def rotate(q, v):
    r = mul(mul(q, (*v, 0)), (-q[0], -q[1], -q[2], q[3]))
    return r[:3]


def angle(a, b):
    d = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
    return math.degrees(2 * math.acos(d))


def clip(path):
    """Local rotations per frame and world positions per frame, by joint name."""
    doc, binary = payload(path)
    nodes = doc['nodes']

    def accessor(index):
        a = doc['accessors'][index]
        v = doc['bufferViews'][a['bufferView']]
        width = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[a['type']]
        offset = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return [struct.unpack_from('<' + 'f' * width, binary, offset + i * width * 4) for i in range(a['count'])]

    anim = doc['animations'][0]
    channels = {}
    for c in anim['channels']:
        channels[c['target']['node'], c['target']['path']] = accessor(anim['samplers'][c['sampler']]['output'])
    frames = len(next(iter(channels.values())))
    parents = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    local, world = [], []
    for f in range(frames):
        out = {}

        def visit(i):
            if i in out:
                return out[i]
            p = channels.get((i, 'translation'), [nodes[i].get('translation', [0, 0, 0])] * frames)[f]
            q = channels.get((i, 'rotation'), [nodes[i].get('rotation', [0, 0, 0, 1])] * frames)[f]
            if i in parents:
                pp, pq = visit(parents[i])
                r = rotate(pq, p)
                p = tuple(pp[k] + r[k] for k in range(3))
                q = mul(pq, q)
            out[i] = (tuple(p), q)
            return out[i]

        for i in range(len(nodes)):
            visit(i)
        names = {i: n['name'] for i, n in enumerate(nodes) if 'name' in n}
        world.append({names[i]: out[i] for i in names})
        local.append({names[i]: channels.get((i, 'rotation'), [nodes[i].get('rotation', [0, 0, 0, 1])] * frames)[f] for i in names})
    return local, world


def measure(path):
    local, world = clip(path)
    n = len(world)
    toes = ['LeftToeBase', 'RightToeBase']
    skate, on = 0.0, 0
    for a, b in zip(world, world[1:]):
        for t in toes:
            if a[t][0][1] < 0.05:
                skate += math.dist(a[t][0], b[t][0]) * FPS
                on += 1
    steps = [max(angle(local[f][j], local[f - 1][j]) for j in local[f]) for f in range(1, n)]
    accel = []
    for f in range(1, n - 1):
        for j in local[f]:
            # Joint angular acceleration proxy: change of consecutive steps (degrees/frame^2).
            accel.append(abs(angle(local[f + 1][j], local[f][j]) - angle(local[f][j], local[f - 1][j])))
    hips = [w['Hips'][0] for w in world]
    return {
        'frames': n,
        'joints': len(local[0]),
        'footSkateToeMetresPerSecond': skate / on if on else None,
        'toeContactSamples': on,
        'jointStepMaxDegrees': max(steps),
        'jointStepMeanDegrees': sum(steps) / len(steps),
        'jointAccelerationMeanDegrees': sum(accel) / len(accel),
        'hips': hips,
        'world': world,
    }


def pose_error(world_a, world_b, joints=None):
    names = joints or [n for n in world_b if n in world_a]
    d = [math.dist(world_a[n][0], world_b[n][0]) for n in names]
    r = [angle(world_a[n][1], world_b[n][1]) for n in names]
    return {'positionMeanMetres': sum(d) / len(d), 'positionMaxMetres': max(d), 'rotationMeanDegrees': sum(r) / len(r), 'rotationMaxDegrees': max(r)}


def main():
    work = Path(sys.argv[1])
    rows = {}
    cache = {}

    def worlds(path):
        if path not in cache:
            cache[path] = clip(path)[1]
        return cache[path]

    for glb in sorted(work.glob('*.glb')):
        recipe_path = Path(str(glb) + '.json')
        if not recipe_path.exists():
            continue
        recipe = json.loads(recipe_path.read_text())
        if recipe.get('command') != 'generate':
            continue
        m = measure(glb)
        row = {k: v for k, v in m.items() if k not in ('hips', 'world')}
        row['correction'] = recipe.get('correction')
        row['rig'] = recipe.get('rig')
        row['kimodoCorrection'] = recipe.get('kimodoCorrection')
        row['densePath'] = recipe.get('densePath')
        row['constraints'] = []
        for c in recipe.get('constraints', []):
            item = {'kind': c['kind'], 'frame': c['frame']}
            if c['kind'] == 'path':
                h = m['hips'][c['frame']]
                item['hipsGroundMetres'] = math.hypot(h[0] - c['x'], h[2] - c['z'])
            elif not c['file'].lower().endswith('.glb'):
                item['independent'] = None  # mapped FBX source: see recipe modelError
                item['recipeModelError'] = c.get('modelError')
            else:
                source = worlds(Path(c['file']))[c['sourceFrame'] % len(worlds(Path(c['file'])))]
                got = m['world'][c['frame']]
                joints = None
                if c.get('effectors'):
                    joints = [{'left-hand': 'LeftHand', 'right-hand': 'RightHand', 'left-foot': 'LeftFoot', 'right-foot': 'RightFoot'}[e] for e in c['effectors'].split(',')]
                # Placement may move or turn a pose; compare in the recipe's frame only when anchored at its file.
                item['independent'] = pose_error(got, source, joints) if c.get('anchored', True) and 'endAt' not in recipe else None
                item['recipeModelError'] = c.get('modelError')
                item['recipeError'] = c.get('error')
            row['constraints'].append(item)
        rows[glb.name] = row
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
