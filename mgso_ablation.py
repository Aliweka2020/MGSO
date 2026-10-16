"""MGSO with on/off switches for every component (for ablation studies).

With all flags True it is the same algorithm as mgso.MGSO (same budget: one evaluate() of `pop` per step).
Flags:
  pbest       True: glide = current-to-pbest/1      False: glide = rand/1 (X[r3] + F*(Xr1-Xr2))
  shrink_elite True: elite pool shrinks 25%->10%    False: fixed 25%
  adapt       True: mF/mCR learned online           False: fixed F~Cauchy(0.5), CR~N(0.8)
  undulate    True: sinusoidal+Levy mode            False: never used
  dive        True: stagnant individuals re-launched around best     False: never
  landing     True: last-phase local search (1/5 rule)               False: never
  binomial    True: dimension-wise crossover        False: all dimensions move (CR=1)
  greedy      True: offspring replaces parent only if not worse      False: offspring always accepted
  leader_protect True: best individual never wanders
"""
import numpy as np
from math import pi
from mgso import MGSO

ALL_ON = dict(pbest=True, shrink_elite=True, adapt=True, undulate=True, dive=True,
              landing=True, binomial=True, greedy=True, leader_protect=True)


class MGSOAblate(MGSO):
    name = "MGSO-ablate"
    flags = dict(ALL_ON)

    def step(s, t):
        fl = s.flags
        P, D, rng = s.pop, s.dim, s.rng
        prog = (t + 1) / s.iters
        order = np.argsort(s.F)
        frac = 0.25 - 0.15 * prog if fl["shrink_elite"] else 0.25
        p = max(2, int(round(P * frac)))
        pb = s.X[order[rng.integers(0, p, P)]]
        Fi = np.clip(s.mF + 0.1 * rng.standard_cauchy(P), 0.05, 1.0)[:, None]
        CRi = np.clip(rng.normal(s.mCR, 0.1, P), 0.0, 1.0)[:, None]
        r1, r2, r3 = rng.integers(0, P, P), rng.integers(0, P, P), rng.integers(0, P, P)
        diff = s.X[r1] - s.X[r2]

        # operators
        if fl["pbest"]:
            glide = s.X + Fi * (pb - s.X) + Fi * diff
        else:
            glide = s.X[r3] + Fi * diff
        wave = np.sin(2 * pi * 3 * prog + rng.uniform(0, 2 * pi, (P, 1)))
        und = s.X + Fi * wave * diff + 0.01 * (1 - prog) * s._levy(P) * s.span
        sig = 0.15 * (1 - prog) ** 2 + 0.002
        dive = s.best_x + sig * (rng.normal(0, 1, (P, D)) + 0.2 * s._levy(P)) * s.span
        land = s.best_x + s.sig_ls * rng.normal(0, 1, (P, D)) * s.span

        # mode assignment
        p_und = (0.05 + 0.35 * (1 - prog)) if fl["undulate"] else 0.0
        mode = np.where(rng.random(P) < p_und, 0, 1)
        if fl["dive"]:
            mode[s.stag >= s.stag_limit] = 2
        if fl["landing"] and prog > s.l_start:
            lm = rng.random(P) < s.l_frac
            mode[lm & (mode != 2)] = 3
        if fl["leader_protect"]:
            mode[order[0]] = 3 if mode[order[0]] == 3 else 1
        V = np.select([mode[:, None] == 0, mode[:, None] == 1, mode[:, None] == 2], [und, glide, dive], land)

        # crossover
        if fl["binomial"]:
            cr = np.where(mode[:, None] >= 2, min(1.0, max(3.0 / D, 0.15)), CRi)
        else:
            cr = np.ones((P, 1))
        mask = rng.random((P, D)) < cr
        mask[np.arange(P), rng.integers(0, D, P)] = True
        U = np.where(mask, V, s.X)
        U = np.where(U < s.lb, (s.lb + s.X) / 2, U); U = np.where(U > s.ub, (s.ub + s.X) / 2, U)
        U = s.clip(U)

        # selection + adaptation
        FU = s.evaluate(U)
        old = s.F.copy()
        better = (FU <= old) if fl["greedy"] else np.ones(P, bool)
        strict = FU < old
        s.X[better], s.F[better] = U[better], FU[better]
        s.stag = np.where(strict, 0, s.stag + 1)
        s.stag[mode == 2] = np.where(strict[mode == 2], 0, s.stag_limit // 2)
        if fl["adapt"]:
            ok = strict & (mode <= 1) & np.isfinite(old - FU)
            if ok.any():
                w = old[ok] - FU[ok]; w = w / (w.sum() + 1e-30)
                f_ok, c_ok = Fi[ok, 0], CRi[ok, 0]
                s.mF = 0.9 * s.mF + 0.1 * float((w * f_ok ** 2).sum() / ((w * f_ok).sum() + 1e-12))
                s.mCR = 0.9 * s.mCR + 0.1 * float((w * c_ok).sum())
        lm = mode == 3
        if lm.any():
            s.sig_ls = float(np.clip(s.sig_ls * (1.3 if strict[lm].mean() > 0.2 else 0.8), 1e-9, 0.1))
        s.mode_log.append([int((mode == k).sum()) for k in range(4)])


def make_variant(name, **off):
    """make_variant('MGSO-noDive', dive=False) -> optimizer class"""
    fl = dict(ALL_ON); fl.update(off)
    return type(name, (MGSOAblate,), dict(name=name, flags=fl))


# core = minimal DE/JADE-like glide only (everything optional switched off)
CORE = dict(shrink_elite=False, adapt=False, undulate=False, dive=False, landing=False, leader_protect=True)

# (1) leave-one-out: full MGSO minus one component
LEAVE_ONE_OUT = {
    "MGSO-full": {},
    "-pbest(rand/1)": dict(pbest=False),
    "-shrink_elite": dict(shrink_elite=False),
    "-adapt": dict(adapt=False),
    "-undulate": dict(undulate=False),
    "-dive": dict(dive=False),
    "-landing": dict(landing=False),
    "-binomial": dict(binomial=False),
    "-greedy": dict(greedy=False),
    "-leader_protect": dict(leader_protect=False),
}

# (2) incremental: start from the core and add one component at a time (cumulative)
INCREMENTAL = {
    "core": dict(CORE),
    "+shrink_elite": dict(CORE, shrink_elite=True),
    "+adapt": dict(CORE, shrink_elite=True, adapt=True),
    "+dive": dict(CORE, shrink_elite=True, adapt=True, dive=True),
    "+landing": dict(CORE, shrink_elite=True, adapt=True, dive=True, landing=True),
    "+undulate(=full)": {},
}

# (3) single-component-only: core + exactly one component (shows standalone value)
ONLY_ONE = {
    "core": dict(CORE),
    "core+adapt": dict(CORE, adapt=True),
    "core+dive": dict(CORE, dive=True),
    "core+landing": dict(CORE, landing=True),
    "core+undulate": dict(CORE, undulate=True),
    "core+shrink": dict(CORE, shrink_elite=True),
}
