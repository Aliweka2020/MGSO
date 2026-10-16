"""Experiment 3 - beam optimisation: PSO/GA/GWO/JAYA/DE/AOA/NiOA on equal evaluation budgets.
problem=angle : refine the steering angle u in [-1,1] (1-D, off-grid beam refinement)
problem=phase : nt analog phase-shifter weights (nt-D, harder; analytic optimum known)"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import add_data_args, steering, RowLogger
from data import get_channels
from optimizers import get_optimizers


def make_problem(h, problem):
    nt = h.size; hc = np.conj(h)
    if problem == "angle":
        lb, ub = np.array([-1.0]), np.array([1.0])
        f = lambda X: -np.abs(hc @ (steering(nt, X[:, 0]) / np.sqrt(nt))) ** 2
        grid = np.linspace(-1, 1, 20001)
        ref = (np.abs(hc @ (steering(nt, grid) / np.sqrt(nt))) ** 2).max()
    else:
        lb, ub = np.full(nt, -np.pi), np.full(nt, np.pi)
        f = lambda X: -np.abs(np.exp(1j * X) @ hc / np.sqrt(nt)) ** 2
        ref = np.abs(h).sum() ** 2 / nt
    return f, lb, ub, ref


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--problem", default="angle", choices=["angle", "phase"])
    ap.add_argument("--opts", default="PSO,GA,GWO,JAYA,DE,AOA,GSO,NiOA")
    ap.add_argument("--instances", type=int, default=50)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--ref_opt", default="NiOA")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ch = get_channels(a); rng = np.random.default_rng(a.seed)
    users = rng.choice(len(ch["H"]), a.instances, replace=False)
    opts = get_optimizers(a.opts.split(","))
    res = {k: np.zeros((a.instances, a.runs)) for k in opts}
    hist = {k: np.zeros(a.iters + 1) for k in opts}
    cbk = []
    log = RowLogger(f"{a.out}/exp3_{a.problem}_runs.csv")
    for i, u in enumerate(users):
        h = ch["H"][u].astype(np.complex128); h /= np.linalg.norm(h)        # ||h|| = 1
        f, lb, ub, ref = make_problem(h, a.problem)
        cb = np.exp(1j * np.pi * np.arange(a.nt)[:, None] * (-1 + 2 * np.arange(a.nbeams)[None, :] / a.nbeams)) / np.sqrt(a.nt)
        cbk.append((np.abs(np.conj(h) @ cb) ** 2).max() / ref)
        for k, cls in opts.items():
            for r in range(a.runs):
                x, fv, hh = cls(f, lb, ub, pop=a.pop, iters=a.iters, seed=1000 * i + r).run()
                res[k][i, r] = -fv / ref; log.add(dict(instance=i, user=int(u), method=k, run=r, gain=-fv / ref)); hist[k] += -hh / ref / (a.instances * a.runs)
    rows = [dict(method="DFT codebook", mean=np.mean(cbk), std=np.std(cbk))]
    for k in opts:
        rows.append(dict(method=k, mean=res[k].mean(), std=res[k].mean(1).std()))
    df = pd.DataFrame(rows); df.to_csv(f"{a.out}/exp3_{a.problem}.csv", index=False)
    print(f"normalised gain (1.0 = optimum), problem={a.problem}, budget={a.pop * (a.iters + 1)} evals/run")
    print(df.round(4).to_string(index=False))
    ref_name = a.ref_opt if a.ref_opt in opts else max(opts, key=lambda k: res[k].mean())
    print(f"\nWilcoxon signed-rank vs {ref_name} (paired over {a.instances} instances):")
    for k in opts:
        if k != ref_name:
            try:
                p = wilcoxon(res[ref_name].mean(1), res[k].mean(1)).pvalue
            except ValueError:
                p = float("nan")
            print(f"  {ref_name} vs {k}: p={p:.4g}")
    np.savez(f"{a.out}/exp3_{a.problem}_convergence.npz", **hist)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for k, v in hist.items():
            plt.plot(v, label=k)
        plt.xlabel("iteration"); plt.ylabel("normalised gain"); plt.legend(); plt.grid(alpha=.3)
        plt.savefig(f"{a.out}/exp3_{a.problem}_convergence.png", dpi=200, bbox_inches="tight")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
