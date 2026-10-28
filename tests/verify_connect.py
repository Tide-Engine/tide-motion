"""Independent dense-GLB verification of connect recipes and retained poses.
Usage: python tests/verify_connect.py SOURCE.glb TARGET.glb OUTPUT.glb
"""
import json
from pathlib import Path
import sys
from verify_cleanup import read, angle


def pins(path, count):
    sidecar = Path(path + '.json')
    if not sidecar.exists():
        return set()
    recipe = json.loads(sidecar.read_text(encoding='utf-8-sig'))
    out = set(recipe.get('protectedFrames', []))
    out.update(c['frame'] for c in recipe.get('constraints', []))
    if recipe.get('command') == 'pose-sequence':
        out.add(0)
        # Early (guide) targets are passed near, not reached exactly.
        out.update(t['endFrame'] for t in recipe['authoring']['targets'] if not t.get('early', 0))
    assert all(isinstance(f, int) and 0 <= f < count for f in out)
    return out


def verify(source, target, output):
    an, a = read(source)
    bn, b = read(target)
    on, out = read(output)
    assert [n['name'] for n in an] == [n['name'] for n in bn] == [n['name'] for n in on]
    na, nb, no = (len(next(iter(t.values()))) for t in (a, b, out))
    recipe = json.loads(Path(output + '.json').read_text())
    offset = recipe['offset']
    def pose(tracks, expected_frame, actual_frame, shifted):
        for (joint, kind), values in tracks.items():
            expected = list(values[expected_frame])
            actual = out[joint, kind][actual_frame]
            if kind == 'rotation':
                assert angle(expected, actual) < 0.002, (joint, actual_frame, 'rotation')
            else:
                if shifted and kind == 'translation' and joint == 0:
                    expected = [v+d for v, d in zip(expected, offset)]
                assert max(abs(x-y) for x, y in zip(expected, actual)) < 2e-6, (joint, actual_frame, kind)
    if recipe['joined']:
        target_start = no - nb
        expected = pins(source, na) | {target_start + f for f in pins(target, nb)} | {na-1, target_start}
        for f in range(na):
            pose(a, f, f, False)
        for f in range(nb):
            pose(b, f, target_start+f, True)
    else:
        expected = {0, no-1}
        pose(a, na-1, 0, False)
        pose(b, 0, no-1, True)
    assert recipe['protectedFrames'] == sorted(expected), (recipe['protectedFrames'], expected)
    print('connected frame protection and all retained poses verified:', no, 'frames, pins', sorted(expected))


if __name__ == '__main__':
    verify(*sys.argv[1:])
