import _bootstrap  # noqa: F401
import numpy as np
import stumpy

rng = np.random.default_rng(0)
normal = rng.normal(size=600)

for length in (6, 18, 36):
    const = np.full(length, 5.0)
    p = stumpy.stump(T_A=normal, m=length, T_B=const, ignore_trivial=False)
    v = np.asarray(p[:, 0], dtype=float)
    finite = v[np.isfinite(v)]
    print(f"L={length}: sqrt(L)={np.sqrt(length):.4f}  observed constant-window distance={finite.min():.4f}")
