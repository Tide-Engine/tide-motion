"""Developer-only independent GLB checks. Uses only the standard library.
No reference motion/checkpoints enter Git; inputs are locally generated clips.
"""
import json
import math
from pathlib import Path
import struct
import sys
from soma_metrics import payload, positions


def tracks(path):
    doc, binary = payload(path)
    result = {}
    for channel in doc['animations'][0]['channels']:
        target = channel['target']
        sampler = doc['animations'][0]['samplers'][channel['sampler']]
        accessor = doc['accessors'][sampler['output']]
        view = doc['bufferViews'][accessor['bufferView']]
        width = {'VEC3': 3, 'VEC4': 4}[accessor['type']]
        offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        result[doc['nodes'][target['node']]['name'], target['path']] = [
            struct.unpack_from('<'+'f'*width, binary, offset+i*width*4)
            for i in range(accessor['count'])]
    return doc, result


def delta(a, b, rotation):
    if rotation and sum(x*y for x, y in zip(a, b)) < 0:
        b = tuple(-x for x in b)
    return max(abs(x-y) for x, y in zip(a, b))


if sys.argv[1] == 'finger-fixture':
    source, output = map(Path, sys.argv[2:4])
    if output.exists():
        raise ValueError('Fixture output already exists')
    raw = bytearray(source.read_bytes())
    doc, _ = payload(source)
    binary_offset = 28 + struct.unpack_from('<I', raw, 12)[0]
    for channel in doc['animations'][0]['channels']:
        target = channel['target']
        if doc['nodes'][target['node']]['name'] != 'LeftHandIndex1' or target['path'] != 'rotation':
            continue
        sampler = doc['animations'][0]['samplers'][channel['sampler']]
        accessor = doc['accessors'][sampler['output']]
        view = doc['bufferViews'][accessor['bufferView']]
        offset = binary_offset+view.get('byteOffset', 0)+accessor.get('byteOffset', 0)
        for frame in range(accessor['count']):
            angle = math.sin(frame*.17)*.4
            struct.pack_into('<4f', raw, offset+frame*16, 0, 0, math.sin(angle/2), math.cos(angle/2))
        output.write_bytes(raw)
        print('Authored varying LeftHandIndex1 fixture:', output)
        break
    else:
        raise ValueError('SOMA-77 fixture lacks the named finger track')
elif sys.argv[1] == 'edit':
    source, edited = map(Path, sys.argv[2:4])
    begin, end = map(int, sys.argv[4:6])
    a_doc, a = tracks(source)
    b_doc, b = tracks(edited)
    assert a.keys() == b.keys(), 'Source rig/tracks changed'
    body = set(tracks(Path(sys.argv[6]))[1]) if len(sys.argv) > 6 else set(a)
    worst = 0
    for key, values in a.items():
        assert len(values) == len(b[key])
        for frame, (x, y) in enumerate(zip(values, b[key])):
            if key not in body or frame < begin or frame >= end or frame in (begin, end-1):
                error = delta(x, y, key[1] == 'rotation')
                worst = max(worst, error)
                assert error < 2e-6, (key, frame, error)
    for frame, (original, world) in enumerate(zip(positions(source), positions(edited))):
        assert all(math.isfinite(v) for p in world.values() for v in p)
        if frame < begin or frame >= end or frame in (begin, end-1):
            for name, point in original.items():
                assert max(abs(x-y) for x, y in zip(point, world[name])) < 2e-5, (name, frame, 'world FK changed')
    print('All outside/end poses and every extra-joint frame preserved; max component error:', worst)
elif sys.argv[1] == 'check':
    folder = Path(sys.argv[2])
    for path in folder.glob('*.glb'):
        doc, values = tracks(path)
        assert len(next(iter(values.values()))) == 89
        assert all(math.isfinite(v) for track in values.values() for row in track for v in row)
        for world in positions(path):
            assert all(math.isfinite(v) for p in world.values() for v in p)
        print(path.name, 'finite local/FK; joints:', len(doc['nodes']))
    cpu = json.loads((folder/'cpu.glb.json').read_text())
    gpu = json.loads((folder/'gpu.glb.json').read_text())
    assert cpu['baseTokens'] == gpu['baseTokens'] and cpu['tokens'] == gpu['tokens']
    _, a = tracks(folder/'gpu.glb')
    _, b = tracks(folder/'gpu-warm.glb')
    assert a == b, 'Same-device warm generation changed'
    _, v = tracks(folder/'variants.seed7.glb')
    for key, values in a.items():
        if key[1] == 'rotation':
            assert max(delta(x, y, True) for x, y in zip(values, v[key])) < 2e-6
    root = v['Hips', 'translation']
    assert math.hypot(root[-1][0]-root[0][0], root[-1][2]-root[0][2]) < 1e-5
    assert gpu['device'] == 'gpu' and cpu['device'] == 'cpu'
    assert json.loads((folder/'cpu-inplace.glb.json').read_text())['threads'] == 1
    assert (folder/'variants.variants.png').is_file()
    print('CPU/GPU full token equality, warm repeat, in-place/rotation and variant controls pass.')
else:
    raise ValueError('Expected finger-fixture, edit or check')
