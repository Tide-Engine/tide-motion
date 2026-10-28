"""Independent FK checks for dart-controls.ps1. Development only, standard library.
No model, generated motion or fixture is committed; the test authors its own clips.
"""
import json
import math
import sys
from pathlib import Path
from dart_metrics import measure
from soma_metrics import positions


def maximum_difference(a, b):
    return max(math.dist(x[j], y[j]) for x, y in zip(a, b) for j in x)


def check(directory):
    root = Path(directory)
    free = positions(root / 'free.glb')
    target = positions(root / 'target.glb')
    errors = {
        'historySeamMetres': maximum_difference(free[-2:], positions(root / 'history.glb')[:2]),
        'sourceSeamMetres': maximum_difference(free[96:98], positions(root / 'source.glb')[:2]),
    }
    # repeat requested a 0.15 m floor projection: test rotations/bone lengths via
    # measure(), and validate its explicitly projected floor rather than root Y.
    assert max(errors.values()) < 1e-4, errors
    endpoints = {}
    for seed in [5, 8]:
        result = positions(root / f'endpose-{seed}.glb')
        endpoints[str(seed)] = maximum_difference([result[-1]], [target[-1]])
        assert endpoints[str(seed)] < .05, endpoints
        assert maximum_difference(result[:2], target[:2]) < 1e-4, 'Changed seed poses'
    for frame in positions(root / 'repeat.glb'):
        assert abs(min(p[1] for p in frame.values()) - .15) < 1e-5, 'Floor projection'
    rows = [measure(p) for p in sorted(root.glob('*.glb'))]
    out = dict(errors, endpointMaximumMetres=endpoints, assets=rows)
    print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    check(sys.argv[1])
