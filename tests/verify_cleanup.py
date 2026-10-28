"""Optional developer check of dense Tide GLB exports; no inference or runtime dependency.
Usage: python tests/verify_cleanup.py BEFORE.glb AFTER.glb LEFT_ANKLE RIGHT_ANKLE
"""
import json
import math
from pathlib import Path
import struct
import sys


def read(path):
    data = Path(path).read_bytes()
    size = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20 + size])
    base = 28 + size
    tracks = {}
    animation = doc['animations'][0]
    for channel in animation['channels']:
        sampler = animation['samplers'][channel['sampler']]
        accessor = doc['accessors'][sampler['output']]
        view = doc['bufferViews'][accessor['bufferView']]
        width = {'VEC3': 3, 'VEC4': 4, 'SCALAR': 1}[accessor['type']]
        offset = base + view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        raw = struct.unpack_from('<' + 'f' * (accessor['count'] * width), data, offset)
        tracks[channel['target']['node'], channel['target']['path']] = [
            raw[i:i + width] for i in range(0, len(raw), width)]
    return doc['nodes'], tracks


def multiply(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return (w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u,
            w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s)


def world(nodes, tracks, frame):
    parents = {child: i for i, node in enumerate(nodes) for child in node.get('children', [])}
    cache = {}

    def visit(i):
        if i in cache:
            return cache[i]
        node = nodes[i]
        def value(kind, default):
            return tracks[i, kind][frame] if (i, kind) in tracks else node.get(kind, default)
        position = value('translation', (0, 0, 0))
        rotation = value('rotation', (0, 0, 0, 1))
        scale = value('scale', (1, 1, 1))[0]
        if i in parents:
            parent_position, parent_rotation, parent_scale = visit(parents[i])
            vector = (*[x * parent_scale for x in position], 0)
            inverse = (*[-x for x in parent_rotation[:3]], parent_rotation[3])
            rotated = multiply(multiply(parent_rotation, vector), inverse)
            position = tuple(parent_position[k] + rotated[k] for k in range(3))
            rotation = multiply(parent_rotation, rotation)
            scale *= parent_scale
        cache[i] = position, rotation, scale
        return cache[i]
    return [visit(i) for i in range(len(nodes))]


def verify(before_path, after_path, left, right):
    nodes, before = read(before_path)
    after_nodes, after = read(after_path)
    assert len(nodes) == len(after_nodes) and before.keys() == after.keys()
    count = len(next(iter(before.values())))
    recipe = json.loads(Path(after_path + '.json').read_text())
    for key in before:
        assert len(before[key]) == len(after[key])
        if key[1] == 'translation' or nodes[key[0]].get('name') == 'Hips':
            assert max(abs(x-y) for a, b in zip(before[key], after[key]) for x, y in zip(a, b)) < 2e-6
        for frame in recipe['protectedFrames']:
            assert max(abs(x-y) for x, y in zip(before[key][frame], after[key][frame])) < 2e-6
    names = {node.get('name'): i for i, node in enumerate(nodes)}
    wa = [world(nodes, before, f) for f in range(count)]
    wb = [world(after_nodes, after, f) for f in range(count)]
    bone_error = 0
    for parent, node in enumerate(nodes):
        for child in node.get('children', []):
            for f in range(count):
                bone_error = max(bone_error, abs(math.dist(wa[f][parent][0], wa[f][child][0]) -
                                                 math.dist(wb[f][parent][0], wb[f][child][0])))
    assert bone_error < 2e-6, bone_error
    for contact in recipe['contacts']:
        joint = names[left if contact['foot'] == 'left-foot' else right]
        start, end = contact['start'], contact['end']
        def drift(worlds):
            points = [worlds[f][joint][0] for f in range(start, end + 1)]
            return sum(math.hypot(b[0]-a[0], b[2]-a[2]) for a, b in zip(points, points[1:])) / (end-start)
        a, b = drift(wa), drift(wb)
        assert abs(a-contact['horizontalMeanStepBeforeMetres']) < 2e-6
        assert abs(b-contact['horizontalMeanStepAfterMetres']) < 2e-6
        for f in range(start, end + 1):
            dot = abs(sum(x*y for x, y in zip(wa[f][joint][1], wb[f][joint][1])))
            assert abs(1-dot) < 2e-5, 'world ankle orientation changed'
        print(contact['foot'], 'mean horizontal step (m):', a, '->', b)
    print('max bone length error (m):', bone_error, '; protected frames:', recipe['protectedFrames'])


if __name__ == '__main__':
    verify(*sys.argv[1:])
