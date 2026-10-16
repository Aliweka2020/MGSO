"""Population-based minimisers with one interface and an equal evaluation budget
(pop * (iters + 1)).  f(X) takes X of shape (pop, dim) and returns (pop,).

NiOA: put your own (published) Ninja Optimization Algorithm implementation in
`nioa_impl.py` as a subclass of Base (see nioa_impl_TEMPLATE.py). It is picked up automatically.
It is intentionally not re-implemented here from memory."""
import numpy as np
from math import gamma, pi, sin


class Base:
    name = "base"

    def __init__(s, f, lb, ub, pop=30, iters=100, seed=0):
        s.f = f; s.lb = np.asarray(lb, float); s.ub = np.asarray(ub, float); s.dim = s.lb.size
        s.pop, s.iters = pop, iters; s.rng = np.random.default_rng(seed)
        s.best_x, s.best_f = None, np.inf

    def clip(s, X):
        return np.clip(X, s.lb, s.ub)

    def evaluate(s, X):
        F = np.asarray(s.f(X), float).reshape(-1)
        i = int(np.argmin(F))
        if F[i] < s.best_f:
            s.best_f, s.best_x = float(F[i]), X[i].copy()
        return F

    def setup(s):
        pass

    def step(s, t):
        raise NotImplementedError

    def run(s):
        s.X = s.lb + s.rng.random((s.pop, s.dim)) * (s.ub - s.lb)
        s.F = s.evaluate(s.X); s.setup()
        hist = [s.best_f]
        for t in range(s.iters):
            s.step(t); hist.append(s.best_f)
        return s.best_x, s.best_f, np.array(hist)


class PSO(Base):
    name = "PSO"

    def setup(s):
        s.V = np.zeros_like(s.X); s.P = s.X.copy(); s.PF = s.F.copy()

    def step(s, t):
        w = 0.9 - 0.5 * t / s.iters
        r1, r2 = s.rng.random((2, s.pop, s.dim))
        s.V = w * s.V + 2.0 * r1 * (s.P - s.X) + 2.0 * r2 * (s.best_x - s.X)
        vmax = 0.2 * (s.ub - s.lb); s.V = np.clip(s.V, -vmax, vmax)
        s.X = s.clip(s.X + s.V); s.F = s.evaluate(s.X)
        b = s.F < s.PF; s.P[b] = s.X[b]; s.PF[b] = s.F[b]


class GA(Base):
    name = "GA"

    def step(s, t):
        P = s.pop; el = max(2, P // 10); order = np.argsort(s.F)
        E, EF = s.X[order[:el]].copy(), s.F[order[:el]].copy(); n = P - el

        def tour():
            c = s.rng.integers(0, P, (n, 3)); return c[np.arange(n), np.argmin(s.F[c], axis=1)]
        p1, p2 = s.X[tour()], s.X[tour()]
        u = s.rng.uniform(-0.25, 1.25, (n, s.dim)); C = p1 + u * (p2 - p1)          # BLX-0.25
        C = np.where(s.rng.random((n, 1)) < 0.9, C, p1)
        m = s.rng.random((n, s.dim)) < 1.0 / s.dim
        C = s.clip(C + m * s.rng.normal(0, 0.1, (n, s.dim)) * (s.ub - s.lb))
        FC = s.evaluate(C)
        s.X, s.F = np.vstack([E, C]), np.r_[EF, FC]


class GWO(Base):
    name = "GWO"

    def step(s, t):
        a = 2 - 2 * t / s.iters; order = np.argsort(s.F); Xn = np.zeros_like(s.X)
        for L in s.X[order[:3]]:
            A = 2 * a * s.rng.random(s.X.shape) - a; C = 2 * s.rng.random(s.X.shape)
            Xn += L - A * np.abs(C * L - s.X)
        s.X = s.clip(Xn / 3); s.F = s.evaluate(s.X)


class JAYA(Base):
    name = "JAYA"

    def step(s, t):
        best, worst = s.X[np.argmin(s.F)], s.X[np.argmax(s.F)]
        r1, r2 = s.rng.random((2, s.pop, s.dim))
        Xn = s.clip(s.X + r1 * (best - np.abs(s.X)) - r2 * (worst - np.abs(s.X)))
        Fn = s.evaluate(Xn); b = Fn < s.F; s.X[b], s.F[b] = Xn[b], Fn[b]


class DE(Base):
    name = "DE"

    def step(s, t):
        Fm, CR = 0.5, 0.9; U = s.X.copy()
        for i in range(s.pop):
            r = s.rng.choice([j for j in range(s.pop) if j != i], 3, replace=False)
            v = s.X[r[0]] + Fm * (s.X[r[1]] - s.X[r[2]])
            m = s.rng.random(s.dim) < CR; m[s.rng.integers(s.dim)] = True
            U[i] = np.where(m, v, s.X[i])
        U = s.clip(U); FU = s.evaluate(U); b = FU < s.F; s.X[b], s.F[b] = U[b], FU[b]


class AOA(Base):
    """Arithmetic Optimization Algorithm (Abualigah et al., 2021), alpha=5, mu=0.499."""
    name = "AOA"

    def step(s, t):
        T = s.iters; MOA = 0.2 + (t + 1) * (1.0 - 0.2) / T
        MOP = 1 - ((t + 1) / T) ** (1 / 5); eps = 1e-12
        r1, r2, r3 = s.rng.random((3, s.pop, s.dim))
        sc = (s.ub - s.lb) * 0.499 + s.lb; b = s.best_x
        Xn = np.where(r1 > MOA,
                      np.where(r2 > 0.5, b / (MOP + eps) * sc, b * MOP * sc),
                      np.where(r3 > 0.5, b - MOP * sc, b + MOP * sc))
        s.X = s.clip(Xn); s.F = s.evaluate(s.X)


class GSO(Base):
    """GSO, ported from the user's reference implementation (sorted population, chain-leader moves).
    Per iteration: sort by fitness; for i = pop-1 .. 1 (worst to best, leader i=0 untouched):
      m ~ U(0,1); if mutation_rate > m and i > pop/2:  x_i = (r1/pop) * x_r1 + A * (F_0/F_i + F_r2/F_r3)
      else:                                             x_i = x_i + A * ((x_0 - x_i) + (x_{i-1} - x_i))
    with r1<=r2<=r3 sorted random indices and A = 1 - t/iters. Updates are sequential, as in the original.
    NOTE: the ratio terms assume positive fitness (benchmark functions). Our objectives are negative
    (-gain, -EE) so the ratios change sign; positive_shift=True evaluates the ratios on F - F_min + 1."""
    name = "GSO"

    def __init__(s, *a, mutation_rate=0.5, positive_shift=False, **k):
        super().__init__(*a, **k); s.mr = mutation_rate; s.shift = positive_shift

    @staticmethod
    def _nz(v):
        return v if abs(v) > 1e-12 else (1e-12 if v >= 0 else -1e-12)

    def step(s, t):
        o = np.argsort(s.F); s.X = s.X[o]; s.F = s.F[o]
        P = s.pop; A = 1 - t / s.iters
        Fr = s.F - s.F[0] + 1.0 if s.shift else s.F
        for i in range(P - 1, 0, -1):
            m = s.rng.random(); r1, r2, r3 = np.sort(s.rng.choice(P, 3, replace=False))
            if s.mr > m and i > 0.5 * P:
                led = Fr[0] / s._nz(Fr[i]); nled = Fr[r2] / s._nz(Fr[r3])
                s.X[i] = (r1 / P) * s.X[r1] + A * (led + nled)
            else:
                s.X[i] = s.X[i] + A * ((s.X[0] - s.X[i]) + (s.X[i - 1] - s.X[i]))
        s.X = s.clip(s.X); s.F = s.evaluate(s.X)




class MGSO(Base):
    name = "MGSO"
    stag_limit = 6          # steps without improvement before an individual dives
    land_start = 0.80       # fraction of run after which landing (local search) starts (earlier if D large)
    land_frac = 0.20        # fraction of population doing landing (larger if D large)

    def setup(s):
        s.mF, s.mCR = 0.5, 0.8
        s.stag = np.zeros(s.pop, int)
        s.span = s.ub - s.lb
        s.sig_ls = 0.02
        hd = min(1.0, s.dim / 64.0)                       # high-dimensional problems get landing earlier / more
        s.l_start = s.land_start - 0.4 * hd
        s.l_frac = s.land_frac + 0.3 * hd
        b = 1.5
        s._lsig = (gamma(1 + b) * sin(pi * b / 2) / (gamma((1 + b) / 2) * b * 2 ** ((b - 1) / 2))) ** (1 / b)
        s.mode_log = []     # (undulate, glide, dive, land) counts per step, used by diagnostics

    def _levy(s, n):
        u = s.rng.normal(0, s._lsig, (n, s.dim)); v = s.rng.normal(0, 1, (n, s.dim))
        return np.clip(u / (np.abs(v) ** (1 / 1.5) + 1e-12), -10, 10)

    def step(s, t):
        P, D, rng = s.pop, s.dim, s.rng
        prog = (t + 1) / s.iters
        order = np.argsort(s.F)
        p = max(2, int(round(P * (0.25 - 0.15 * prog))))                   # shrinking elite pool
        pb = s.X[order[rng.integers(0, p, P)]]
        Fi = np.clip(s.mF + 0.1 * rng.standard_cauchy(P), 0.05, 1.0)[:, None]
        CRi = np.clip(rng.normal(s.mCR, 0.1, P), 0.0, 1.0)[:, None]
        r1, r2 = rng.integers(0, P, P), rng.integers(0, P, P)
        diff = s.X[r1] - s.X[r2]

        # --- operators -------------------------------------------------------------------
        glide = s.X + Fi * (pb - s.X) + Fi * diff
        wave = np.sin(2 * pi * 3 * prog + rng.uniform(0, 2 * pi, (P, 1)))
        und = s.X + Fi * wave * diff + 0.01 * (1 - prog) * s._levy(P) * s.span
        sig = 0.15 * (1 - prog) ** 2 + 0.002
        dive = s.best_x + sig * (rng.normal(0, 1, (P, D)) + 0.2 * s._levy(P)) * s.span
        land = s.best_x + s.sig_ls * rng.normal(0, 1, (P, D)) * s.span

        # --- mode assignment ---------------------------------------------------------------
        mode = np.where(rng.random(P) < 0.05 + 0.35 * (1 - prog), 0, 1)   # 0 undulate, 1 glide
        mode[s.stag >= s.stag_limit] = 2                                   # dive
        if prog > s.l_start:
            lm = rng.random(P) < s.l_frac
            mode[lm & (mode != 2)] = 3                                     # landing
        mode[order[0]] = np.where(mode[order[0]] == 3, 3, 1)               # leader never wanders
        V = np.select([mode[:, None] == 0, mode[:, None] == 1, mode[:, None] == 2], [und, glide, dive], land)

        # --- crossover mask (dimension-wise) -----------------------------------------------
        cr = np.where(mode[:, None] >= 2, min(1.0, max(3.0 / D, 0.15)), CRi)
        mask = rng.random((P, D)) < cr
        mask[np.arange(P), rng.integers(0, D, P)] = True
        U = np.where(mask, V, s.X)
        U = np.where(U < s.lb, (s.lb + s.X) / 2, U); U = np.where(U > s.ub, (s.ub + s.X) / 2, U)
        U = s.clip(U)

        # --- selection + adaptation --------------------------------------------------------
        FU = s.evaluate(U)
        old = s.F.copy()
        better = FU <= old
        strict = FU < old
        s.X[better], s.F[better] = U[better], FU[better]
        s.stag = np.where(strict, 0, s.stag + 1)
        s.stag[mode == 2] = np.where(strict[mode == 2], 0, s.stag_limit // 2)   # give dives a 2nd chance window
        ok = strict & (mode <= 1) & np.isfinite(old - FU)
        if ok.any():
            w = old[ok] - FU[ok]; w = w / (w.sum() + 1e-30)
            f_ok, c_ok = Fi[ok, 0], CRi[ok, 0]
            s.mF = 0.9 * s.mF + 0.1 * float((w * f_ok ** 2).sum() / ((w * f_ok).sum() + 1e-12))
            s.mCR = 0.9 * s.mCR + 0.1 * float((w * c_ok).sum())
        lm = mode == 3
        if lm.any():                                                        # 1/5 success rule
            s.sig_ls = float(np.clip(s.sig_ls * (1.3 if strict[lm].mean() > 0.2 else 0.8), 1e-9, 0.1))
        s.mode_log.append([int((mode == k).sum()) for k in range(4)])


OPTIMIZERS = {c.name: c for c in (PSO, GA, GWO, JAYA, DE, AOA, GSO,MGSO)}
try:
    from nioa_impl import NiOA          # your implementation (subclass of Base, name="NiOA")
    OPTIMIZERS["NiOA"] = NiOA
except ImportError:
    pass


def get_optimizers(names):
    out = {}
    for n in names:
        if n in OPTIMIZERS:
            out[n] = OPTIMIZERS[n]
        else:
            print(f"[warn] optimizer '{n}' not available (NiOA needs nioa_impl.py) -> skipped")
    return out
