"""Modified Glider Snake Optimizer (MGSO) - proposed algorithm (subclass of optimizers.Base).

Metaphor: a flying snake climbs (explores), undulates its body while airborne, glides toward the
landing zone, and finally makes a small controlled "landing" adjustment.  Each metaphor maps to a
concrete operator, and each operator targets a known weakness of metaheuristics:

 Failure mode                      MGSO remedy
 --------------------------------  ----------------------------------------------------------
 premature convergence             UNDULATION: sinusoidal difference-vector moves + Levy jumps,
                                   probability decays with progress (kept >0 until the end)
 stagnation / lost individuals     DIVE: an individual with no improvement for `stag_limit`
                                   steps is re-launched around the elite (Gaussian+Levy) -> no
                                   dead population members
 hand-tuned control parameters     GLIDE: current-to-pbest/1 move with success-history adapted
                                   F and CR (learned online, no manual tuning)
 poor scaling to high dimensions   binomial dimension-mask crossover (only some coordinates move)
 no fine convergence               LANDING: last 20 % of iterations, a slice of the population does
                                   local search around the best with 1/5-success-rule step size
 population degradation            greedy elitist selection (offspring replaces parent only if
                                   not worse)

Budget: exactly one evaluate() call of `pop` candidates per step -> same budget as the others.
"""
import numpy as np
from math import gamma, pi, sin
from optimizers import Base


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
