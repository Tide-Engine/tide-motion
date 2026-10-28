"""Developer-only parity check of the native dense-path smoother.

Transcribes NVIDIA Kimodo's kimodo/motion_rep/smooth_root.py (Apache-2.0,
nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954) with NumPy only: the sparse
LU solve becomes a dense solve. Not used by the tool, its build or its CTests.

python tests/smooth_reference.py CAPABILITY_TESTS.exe
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np


def smoother_solve(margins, pos_weight=0.0, alpha=1.8, iters=500):
    n = len(margins)
    a = np.zeros((n, n))
    for i in range(1, n - 1):
        a[i, i - 1:i + 2] = [-1.0, 2.0, -1.0]
    m = pos_weight * np.eye(n) + a.T @ a
    step = 0.25 * np.sqrt(np.abs(np.diag(m)).max())
    m = m + step * np.eye(n)

    def smooth(targets, x0):
        x = x0.copy()
        z = np.zeros_like(x)
        u = np.zeros_like(x)
        for _ in range(iters):
            z[:] = x + u - targets
            norms = np.linalg.norm(z, axis=1)
            mask = norms > margins
            if np.any(mask):
                z[mask] *= (margins[mask] / norms[mask])[:, None]
            z[:] += targets
            u[:] += alpha * (x - z)
            x[:] = np.linalg.solve(m, pos_weight * targets + step * (z - u))
        return x

    return smooth


def smooth_signal(x, margins):
    xs = x.copy()
    xs[:] = x.mean(axis=0, keepdims=True)
    levels = max(int(math.floor(math.log2(len(x)))) - 4, 1)
    stepsize = 2 ** levels
    while True:
        num_steps = len(xs[::stepsize])
        xs[::stepsize] = smoother_solve(margins[::stepsize])(x[::stepsize], xs[::stepsize])
        nxt = stepsize // 2
        num_inter = len(xs[nxt::stepsize])
        if num_inter == num_steps:
            xs[nxt::stepsize][-1] = xs[::stepsize][-1] + (xs[::stepsize][-1] - xs[::stepsize][-2]) / 2
            num_inter -= 1
        xs[nxt::stepsize][:num_inter] = (xs[::stepsize][:-1] + xs[::stepsize][1:]) / 2
        if stepsize == 1:
            break
        stepsize //= 2
    return xs


def cases():
    t = np.arange(90, dtype=np.float64)
    yield "line", np.stack([0.04 * t, 0.01 * t], 1)
    corner = [[0.05 * i, 0.0] if i < 30 else [1.5, 0.05 * (i - 30)] for i in range(60)]
    yield "corner", np.array(corner)
    rng = np.random.default_rng(7)
    yield "wander", np.cumsum(rng.normal(0, 0.03, (137, 2)), axis=0)
    yield "short", np.array([[0, 0], [0.1, 0], [0.2, 0.05], [0.3, 0.1], [0.4, 0.1], [0.5, 0.2]], dtype=float)


def main():
    tool = sys.argv[1]
    worst = 0.0
    with tempfile.TemporaryDirectory() as d:
        for name, x in cases():
            src, dst = os.path.join(d, name + ".csv"), os.path.join(d, name + ".out.csv")
            np.savetxt(src, x, delimiter=",", fmt="%.17g")
            subprocess.run([tool, "--smooth", src, dst], check=True)
            native = np.loadtxt(dst, delimiter=",", ndmin=2)
            ref = smooth_signal(x, np.full(len(x), 0.06))
            err = np.abs(native - ref).max()
            worst = max(worst, err)
            print(f"{name}: {len(x)} frames, max |native - reference| = {err:.3e} m")
    print(f"worst {worst:.3e} m")
    sys.exit(0 if worst < 1e-9 else 1)


if __name__ == "__main__":
    main()
