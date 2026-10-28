"""Regenerates src/soma77.hpp from an NVIDIA Kimodo checkout (developer tool only).

python tools/generate_soma77.py PATH/TO/kimodo [src/soma77.hpp]

Reads, at nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 (Apache-2.0):
kimodo/skeleton/definitions.py (SOMASkeleton77 names/parents),
kimodo/assets/skeletons/somaskel77/joints.p (global neutral joints, a torch zip
pickle read here with a tensor-only unpickler; no torch) and
relaxed_hands_rest_pose.npy. Needs NumPy. Not shipped, built or run by the tool.
"""
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np


def load_tensor(path):
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split('/')[0]

    class Storage:
        def __init__(self, dtype, key):
            self.dtype, self.key = dtype, key

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                def rebuild(storage, offset, size, stride, *rest):
                    raw = np.frombuffer(archive.read(f'{prefix}/data/{storage.key}'), dtype=storage.dtype)
                    return np.lib.stride_tricks.as_strided(raw[offset:], shape=size, strides=[s * raw.itemsize for s in stride]).copy()
                return rebuild
            if module == 'torch' and name in ('FloatStorage', 'DoubleStorage'):
                return {'FloatStorage': np.float32, 'DoubleStorage': np.float64}[name]
            if module == 'collections' and name == 'OrderedDict':
                import collections
                return collections.OrderedDict
            raise pickle.UnpicklingError(f'refusing {module}.{name}')

        def persistent_load(self, pid):
            return Storage(pid[1], pid[2])

    return Reader(archive.open(f'{prefix}/data.pkl')).load()


def literal(v):
    text = '%.9g' % v
    if text in ('0', '-0'):
        return '0.f'
    if '.' not in text and 'e' not in text:
        text += '.'
    return text + 'f'


def quaternion(m):
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4])
    else:
        i = int(np.argmax(np.diag(m))); j = (i + 1) % 3; k = (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        q = np.zeros(4); q[i] = s / 4; q[j] = (m[j, i] + m[i, j]) / s; q[k] = (m[k, i] + m[i, k]) / s; q[3] = (m[k, j] - m[j, k]) / s
    if q[3] < 0:
        q = -q
    return q / np.linalg.norm(q)


def wrap(items, indent='    ', width=150):
    lines, current = [], ''
    for k, item in enumerate(items):
        piece = item + (',' if k + 1 < len(items) else '')
        if current and len(current) + len(piece) > width:
            lines.append(current)
            current = indent
        current += piece
    lines.append(current)
    return '\n'.join(lines)


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / 'src' / 'soma77.hpp'
    source = (root / 'kimodo/skeleton/definitions.py').read_text()
    block = source.split('class SOMASkeleton77')[1].split('bone_order_names_with_parents = [')[1].split(']\n')[0]
    pairs = re.findall(r'\("(\w+)", (?:"(\w+)"|None)\)', block)
    index = {n: i for i, (n, _) in enumerate(pairs)}
    parents = [-1 if p == '' else index[p] for _, p in pairs]
    assets = root / 'kimodo/assets/skeletons/somaskel77'
    joints = load_tensor(assets / 'joints.p').astype(np.float64)
    offsets = np.array([joints[i] - (joints[parents[i]] if parents[i] >= 0 else 0) for i in range(len(pairs))])
    relaxed = np.load(assets / 'relaxed_hands_rest_pose.npy').astype(np.float64)
    for m in relaxed:
        assert abs(np.linalg.det(m) - 1) < 1e-4 and np.abs(m @ m.T - np.eye(3)).max() < 1e-4
    quats = [quaternion(m) for m in relaxed]
    reflect = np.diag([-1, 1, 1])
    asym_offset = asym_rotation = 0.0
    for i, (n, _) in enumerate(pairs):
        if n.startswith('Left'):
            j = index['Right' + n[4:]]
            asym_offset = max(asym_offset, np.abs(offsets[i] * [-1, 1, 1] - offsets[j]).max())
            asym_rotation = max(asym_rotation, np.abs(reflect @ relaxed[i] @ reflect - relaxed[j]).max())
    print(f'left/right asymmetry: offsets {asym_offset:.4g} m, relaxed rotations {asym_rotation:.4g}', file=sys.stderr)
    text = '\n'.join([
        '// Generated from NVIDIA Kimodo (Apache-2.0, nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954):',
        '// joint names and parents from kimodo/skeleton/definitions.py SOMASkeleton77, parent-local',
        '// rest offsets from kimodo/assets/skeletons/somaskel77/joints.p (global neutral joints),',
        '// relaxed rotations from relaxed_hands_rest_pose.npy as unit XYZW quaternions.',
        '// Reproduce with tools/generate_soma77.py; do not edit by hand.',
        '#pragma once', '#include <array>', '#include <string_view>', 'namespace motion::soma77_data {',
        'inline constexpr std::array<std::string_view,77> names{' + wrap([f'"{n}"' for n, _ in pairs]) + '};',
        'inline constexpr std::array<int,77> parents{' + wrap([str(p) for p in parents]) + '};',
        'inline constexpr std::array<std::array<float,3>,77> offsets{{' + wrap(['{' + ','.join(literal(v) for v in o) + '}' for o in offsets]) + '}};',
        'inline constexpr std::array<std::array<float,4>,77> relaxed{{' + wrap(['{' + ','.join(literal(v) for v in q) + '}' for q in quats]) + '}};',
        '}', ''])
    out.write_text(text, newline='\n')


if __name__ == '__main__':
    main()
