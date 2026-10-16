"""Glider Snake Optimizer (GSO) - adapter of the ORIGINAL function-style GSO into the optimizers.Base
interface so it runs in exp2/3/4/5/6 with the same evaluation budget as the other methods.

Update rules are copied 1:1 from the original code:
  * population sorted by fitness each iteration, leader (index 0) is never moved (elitism)
  * A = 1 - t/iters
  * for s = pop-1 .. 1, one random m in [0,1) per solution, rs1<=rs2<=rs3 = sorted random indices
      if mutation_rate > m and s > 0.5*pop:  x_s = (rs1/pop)*x_rs1 + A*(F_leader/F_s + F_rs2/F_rs3)
      else:                                  x_s = x_s + A*((x_leader - x_s) + (x_{s-1} - x_s))
Differences forced by the framework (not by the algorithm):
  * bounds are clipped right after the update, then all `pop` candidates are evaluated once per step
    (original evaluates at the start of each iteration -> identical except one extra final evaluation)
  * fitness ratios use a tiny epsilon guard against division by ~0 (original would give inf/nan)
"""
import numpy as np
from optimizers import Base


def _nz(v, eps=1e-12):
    v = np.asarray(v, float)
    return np.where(np.abs(v) < eps, np.where(v < 0, -eps, eps), v)


class GSO(Base):
    name = "GSO"
    mutation_rate = 0.5

    def step(s, t):
        o = np.argsort(s.F); s.X = s.X[o]; s.F = s.F[o]
        P = s.pop; lead = s.F[0]; A = 1 - t / s.iters
        for i in range(P - 1, 0, -1):
            m = s.rng.random()
            rs1, rs2, rs3 = np.sort(s.rng.choice(P, 3, replace=False))
            if s.mutation_rate > m and i > 0.5 * P:
                s.X[i] = (rs1 / P) * s.X[rs1] + A * (lead / _nz(s.F[i]) + s.F[rs2] / _nz(s.F[rs3]))
            else:
                sd = s.X[i].copy()
                s.X[i] = sd + A * ((s.X[0] - sd) + (s.X[i - 1] - sd))
        s.X = s.clip(s.X); s.F = s.evaluate(s.X)
