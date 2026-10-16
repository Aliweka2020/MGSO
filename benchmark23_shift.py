"""The 23 classical benchmark functions (F1-F23, Yao et al. 1999 / used in GWO, SSA, WOA papers).

Every function is vectorised: f(X) with X of shape (n, d) -> array of shape (n,)  (minimisation).
F1-F7 unimodal, F8-F13 multimodal (scalable dimension), F14-F23 fixed-dimension multimodal.
get_function(name, dim) -> f, lb, ub, d, f_min
"""
import numpy as np
from numpy import pi, sin, cos, exp, sqrt


def _u(x, a, k, m):
    return np.where(x > a, k * (x - a) ** m, np.where(x < -a, k * (-x - a) ** m, 0.0))


# ----------------------------------------------------------------- unimodal
def F1(X): return (X ** 2).sum(1)
def F2(X): A = np.abs(X); return A.sum(1) + A.prod(1)
def F3(X): return (np.cumsum(X, 1) ** 2).sum(1)
def F4(X): return np.abs(X).max(1)
def F5(X): return (100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2).sum(1)
def F6(X): return (np.floor(X + 0.5) ** 2).sum(1)
def F7(X): return (np.arange(1, X.shape[1] + 1) * X ** 4).sum(1) + np.random.random(X.shape[0])   # noisy


# --------------------------------------------------------------- multimodal
def F8(X): return -(X * sin(sqrt(np.abs(X)))).sum(1)
def F9(X): return (X ** 2 - 10 * cos(2 * pi * X) + 10).sum(1)
def F10(X): return -20 * exp(-0.2 * sqrt((X ** 2).mean(1))) - exp(cos(2 * pi * X).mean(1)) + 20 + np.e
def F11(X): return (X ** 2).sum(1) / 4000 - np.prod(cos(X / sqrt(np.arange(1, X.shape[1] + 1))), 1) + 1


def F12(X):
    d = X.shape[1]; y = 1 + (X + 1) / 4
    t = 10 * sin(pi * y[:, 0]) ** 2 + ((y[:, :-1] - 1) ** 2 * (1 + 10 * sin(pi * y[:, 1:]) ** 2)).sum(1) + (y[:, -1] - 1) ** 2
    return pi / d * t + _u(X, 10, 100, 4).sum(1)


def F13(X):
    t = sin(3 * pi * X[:, 0]) ** 2 + ((X[:, :-1] - 1) ** 2 * (1 + sin(3 * pi * X[:, 1:]) ** 2)).sum(1) \
        + (X[:, -1] - 1) ** 2 * (1 + sin(2 * pi * X[:, -1]) ** 2)
    return 0.1 * t + _u(X, 5, 100, 4).sum(1)


# ---------------------------------------------------------- fixed dimension
_g = np.array([-32, -16, 0, 16, 32])
_A14 = np.array([np.tile(_g, 5), np.repeat(_g, 5)], float)                     # (2, 25)
def F14(X):
    s = ((X[:, :, None] - _A14[None]) ** 6).sum(1)                              # (n, 25)
    return 1.0 / (1 / 500 + (1.0 / (np.arange(1, 26) + s)).sum(1))


_aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
_bK = 1.0 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
def F15(X):
    x1, x2, x3, x4 = (X[:, i:i + 1] for i in range(4))
    return ((_aK - x1 * (_bK ** 2 + x2 * _bK) / (_bK ** 2 + x3 * _bK + x4)) ** 2).sum(1)


def F16(X):
    x, y = X[:, 0], X[:, 1]
    return 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4


def F17(X):
    x, y = X[:, 0], X[:, 1]
    return (y - 5.1 / (4 * pi ** 2) * x ** 2 + 5 / pi * x - 6) ** 2 + 10 * (1 - 1 / (8 * pi)) * cos(x) + 10


def F18(X):
    x, y = X[:, 0], X[:, 1]
    a = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
    b = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    return a * b


_cH = np.array([1, 1.2, 3, 3.2])
_a3 = np.array([[3, 10, 30], [.1, 10, 35], [3, 10, 30], [.1, 10, 35]], float)
_p3 = np.array([[.3689, .117, .2673], [.4699, .4387, .747], [.1091, .8732, .5547], [.03815, .5743, .8828]])
_a6 = np.array([[10, 3, 17, 3.5, 1.7, 8], [.05, 10, 17, .1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, .05, 10, .1, 14]])
_p6 = np.array([[.1312, .1696, .5569, .0124, .8283, .5886], [.2329, .4135, .8307, .3736, .1004, .9991],
                [.2348, .1451, .3522, .2883, .3047, .6650], [.4047, .8828, .8732, .5743, .1091, .0381]])
def _hart(X, a, p): return -(_cH * exp(-(a[None] * (X[:, None, :] - p[None]) ** 2).sum(2))).sum(1)
def F19(X): return _hart(X, _a3, _p3)
def F20(X): return _hart(X, _a6, _p6)


_aS = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7],
                [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]], float)
_cS = np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
def _shekel(X, m): return -(1.0 / (((X[:, None, :] - _aS[None, :m]) ** 2).sum(2) + _cS[:m])).sum(1)
def F21(X): return _shekel(X, 5)
def F22(X): return _shekel(X, 7)
def F23(X): return _shekel(X, 10)


# name: (function, lower, upper, dimension (None = user chosen), known optimum value or callable(d))
_REG = {
    "F1": (F1, -100, 100, None, 0), "F2": (F2, -10, 10, None, 0), "F3": (F3, -100, 100, None, 0),
    "F4": (F4, -100, 100, None, 0), "F5": (F5, -30, 30, None, 0), "F6": (F6, -100, 100, None, 0),
    "F7": (F7, -1.28, 1.28, None, 0),
    "F8": (F8, -500, 500, None, lambda d: -418.9829 * d), "F9": (F9, -5.12, 5.12, None, 0),
    "F10": (F10, -32, 32, None, 0), "F11": (F11, -600, 600, None, 0),
    "F12": (F12, -50, 50, None, 0), "F13": (F13, -50, 50, None, 0),
    "F14": (F14, -65.536, 65.536, 2, 0.998004), "F15": (F15, -5, 5, 4, 0.0003075),
    "F16": (F16, -5, 5, 2, -1.031628),
    "F17": (F17, [-5, 0], [10, 15], 2, 0.397887),            # standard Branin domain
    "F18": (F18, -2, 2, 2, 3.0), "F19": (F19, 0, 1, 3, -3.86278), "F20": (F20, 0, 1, 6, -3.32237),
    "F21": (F21, 0, 10, 4, -10.1532), "F22": (F22, 0, 10, 4, -10.4028), "F23": (F23, 0, 10, 4, -10.5363),
}
GROUPS = {"unimodal": [f"F{i}" for i in range(1, 8)], "multimodal": [f"F{i}" for i in range(8, 14)],
          "fixed": [f"F{i}" for i in range(14, 24)], "all": list(_REG)}


# coordinate of the global optimum in the ORIGINAL functions (0 for all the others)
_XOPT = {"F5": 1.0, "F8": 420.9687, "F12": -1.0, "F13": 1.0}


def _rot(rng, d):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))                       # random orthogonal matrix


def get_function(name, dim=30, shift=False, rotate=False, shift_frac=0.4, seed=0):
    """Returns f, lb, ub, d, f_min.

    shift=True  : the optimum is moved to a random point p (a DIFFERENT random value per coordinate, drawn within
                  +-shift_frac of the half-range around the centre, and kept >=5% of the range away from the bounds).
    rotate=True : the search space is rotated by a random orthogonal matrix around the optimum (breaks separability).
    g(x) = f((x - p) @ M + x_opt)  ->  g(p) = f_min.  Applied to the scalable functions F1-F13 only; F14-F23 are
    low-dimensional with specific optimum locations and are left unchanged.
    The same shift/rotation is used for all optimizers and runs (fixed by `seed`).
    Notes: F1 (sphere) is rotation invariant. For shifted F8 the landscape extends beyond +-500, so its true
    minimum can be lower than -418.98*d: compare algorithms with each other, not with the original optimum.
    """
    f, lo, hi, d, fmin = _REG[name]
    scalable = d is None
    d = dim if scalable else d
    lb = np.broadcast_to(np.asarray(lo, float), (d,)).copy()
    ub = np.broadcast_to(np.asarray(hi, float), (d,)).copy()
    fm = fmin(d) if callable(fmin) else fmin
    if not (shift or rotate) or not scalable:
        return f, lb, ub, d, fm

    rng = np.random.default_rng([seed, int(name[1:])])
    xopt = _XOPT.get(name, 0.0)
    mid, half, span = (lb + ub) / 2, (ub - lb) / 2, ub - lb
    if shift:
        p = np.clip(mid + rng.uniform(-1, 1, d) * shift_frac * half, lb + 0.05 * span, ub - 0.05 * span)
    else:
        p = np.full(d, xopt)
    M = _rot(rng, d) if rotate else None

    def g(X):
        Z = (X - p) if M is None else (X - p) @ M
        return f(Z + xopt)

    g.x_opt = p
    return g, lb, ub, d, fm


if __name__ == "__main__":      # sanity check: evaluate each function at its known optimum (approx.)
    for n in _REG:
        f, lb, ub, d, fmin = get_function(n, 30)
        pts = {"F5": np.ones(d), "F8": np.full(d, 420.9687), "F12": np.full(d, -1.0), "F13": np.ones(d),
               "F14": np.array([-32., -32]), "F15": np.array([.19, .19, .12, .14]),
               "F16": np.array([.0898, -.7126]), "F17": np.array([pi, 2.275]), "F18": np.array([0., -1]),
               "F19": np.array([.114, .556, .852]), "F20": np.array([.201, .15, .477, .275, .312, .657]),
               "F21": np.full(4, 4.0), "F22": np.full(4, 4.0), "F23": np.full(4, 4.0)}.get(n, np.zeros(d))
        print(f"{n:4s} d={d:2d}  f(x*)={f(pts[None])[0]: .5f}   expected {fmin: .5f}")
    print("\nshifted+rotated check: g(x_opt) should equal f_min (F7 has +U[0,1) noise)")
    for n in GROUPS["unimodal"] + GROUPS["multimodal"]:
        g, lb, ub, d, fmin = get_function(n, 30, shift=True, rotate=True, seed=0)
        inside = bool(((g.x_opt >= lb) & (g.x_opt <= ub)).all())
        print(f"{n:4s} g(x_opt)={g(g.x_opt[None])[0]: .5f}   f_min {fmin: .5f}   optimum inside bounds: {inside}")
