"""Run MGSO and other optimizers on the 23 classical benchmark functions -> Mean/STD table.

python exp_bench23.py --opts PSO,GA,GWO,JAYA,DE,AOA,GSO,MGSO --dim 30 --pop 30 --iters 500 --runs 30
python exp_bench23.py --funcs F1,F9,F14 --runs 5 --iters 100            # quick test
python exp_bench23.py --funcs unimodal                                  # groups: unimodal|multimodal|fixed|all
Extra optimizer name: MGSO-noUnd (MGSO without undulation, needs mgso_ablation.py).
All optimizers get the same budget pop*(iters+1) and the same seeds (1000*function_index + run).
"""
import argparse, os, time
import numpy as np, pandas as pd
from scipy.stats import ranksums, rankdata
from benchmark23 import GROUPS, get_function
from optimizers import get_optimizers


class Counter:
    """Wraps the objective and counts how many candidate solutions (rows) were actually evaluated (FEs)."""
    def __init__(s, f):
        s.f, s.n = f, 0

    def __call__(s, X):
        X = np.asarray(X)
        s.n += 1 if X.ndim == 1 else X.shape[0]
        return s.f(X)


def build(names):
    algos = {}
    for n in names:
        if n == "MGSO-noUnd":
            from mgso_ablation import make_variant
            algos[n] = make_variant(n, undulate=False)
        else:
            algos.update(get_optimizers([n]))
    return algos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funcs", default="all")
    ap.add_argument("--opts", default="PSO,GA,GWO,JAYA,DE,AOA,GSO,MGSO")
    ap.add_argument("--dim", type=int, default=30, help="dimension of F1-F13 (F14-F23 have fixed dimension)")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--ref", default="MGSO", help="reference method for the rank-sum test")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plots", action="store_true", help="save convergence curves")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    funcs = GROUPS[a.funcs] if a.funcs in GROUPS else a.funcs.split(",")
    algos = build(a.opts.split(","))
    ref = a.ref if a.ref in algos else list(algos)[-1]
    raw, hist, rows, tim, fes = {}, {}, [], {}, {}

    for fi, fn in enumerate(funcs):
        f, lb, ub, d, fmin = get_function(fn, a.dim)
        for k, cls in algos.items():
            vals, hs = np.zeros(a.runs), []
            ts, fe = np.zeros(a.runs), np.zeros(a.runs)
            for r in range(a.runs):
                cf = Counter(f)
                t0 = time.perf_counter()
                x, fv, hh = cls(cf, lb, ub, pop=a.pop, iters=a.iters, seed=1000 * fi + r).run()
                ts[r] = time.perf_counter() - t0; fe[r] = cf.n
                vals[r] = fv; hs.append(np.asarray(hh, float))
                rows.append(dict(function=fn, method=k, run=r, best=fv, time_s=ts[r], FEs=fe[r]))
            raw[fn, k] = vals; hist[fn, k] = np.mean(hs, 0); tim[fn, k] = ts; fes[fn, k] = fe
        print(f"{fn} (d={d}) done", flush=True)

    # ---- Mean / STD table
    tab = pd.DataFrame({(k, m): [fn_stat(raw[fn, k], m) for fn in funcs] for k in algos for m in ("Mean", "STD")},
                       index=funcs)
    tab.columns = pd.MultiIndex.from_tuples(tab.columns)
    tab.to_csv(f"{a.out}/bench23_mean_std.csv")
    pd.DataFrame(rows).to_csv(f"{a.out}/bench23_runs.csv", index=False)

    # ---- time and function-evaluation tables
    cols = {}
    for k in algos:
        cols[k, "Avg time (s)"] = [tim[fn, k].mean() for fn in funcs]
        cols[k, "STD time (s)"] = [tim[fn, k].std(ddof=1) if a.runs > 1 else 0.0 for fn in funcs]
        cols[k, "Avg FEs"] = [fes[fn, k].mean() for fn in funcs]
    ttab = pd.DataFrame(cols, index=funcs)
    ttab.columns = pd.MultiIndex.from_tuples(ttab.columns)
    ttab.to_csv(f"{a.out}/bench23_time_fes.csv")
    print(f"\nAverage time, STD of time and average FEs per run (nominal budget = {a.pop * (a.iters + 1)} FEs)")
    print(ttab.to_string(float_format=lambda v: f"{v:.4g}"))
    summ_t = pd.DataFrame({k: dict(avg_time_over_functions=np.mean([tim[fn, k].mean() for fn in funcs]),
                                   total_time_s=sum(tim[fn, k].sum() for fn in funcs),
                                   avg_FEs=np.mean([fes[fn, k].mean() for fn in funcs])) for k in algos}).T
    summ_t.to_csv(f"{a.out}/bench23_time_summary.csv")
    print("\nOverall (averaged over all selected functions):")
    print(summ_t.to_string(float_format=lambda v: f"{v:.4g}"))
    if any(abs(fes[fn, k].mean() - a.pop * (a.iters + 1)) > 0.5 for fn in funcs for k in algos):
        print("[note] some methods used a number of FEs different from pop*(iters+1): budgets are NOT equal.")
    print(f"\nMean and STD of best fitness, dim={a.dim}, pop={a.pop}, iters={a.iters}, runs={a.runs}, "
          f"budget={a.pop * (a.iters + 1)} evals/run (lower is better)")
    print(tab.to_string(float_format=lambda v: f"{v:.3e}"))

    # ---- text table with the best mean marked by '*'
    means = np.array([[raw[fn, k].mean() for k in algos] for fn in funcs])
    stds = np.array([[raw[fn, k].std(ddof=1) if a.runs > 1 else 0.0 for k in algos] for fn in funcs])
    txt = pd.DataFrame([[f"{means[i, j]:.3e} ({stds[i, j]:.2e})" + ("*" if means[i, j] == means[i].min() else "")
                         for j in range(len(algos))] for i in range(len(funcs))], index=funcs, columns=list(algos))
    txt.to_csv(f"{a.out}/bench23_mean_std_text.csv")
    try:
        with open(f"{a.out}/bench23_mean_std.tex", "w") as fh:
            fh.write(txt.to_latex())
    except Exception as e:
        print("latex export skipped:", e)

    # ---- average rank (Friedman-style) and Wilcoxon rank-sum vs ref
    ranks = np.array([rankdata(m) for m in means])
    print("\nAverage rank over functions (1 = best):")
    print(pd.Series(ranks.mean(0), index=list(algos)).sort_values().round(3).to_string())
    print(f"\nWilcoxon rank-sum (alpha=0.05) of {ref} vs others, over {a.runs} runs per function  [ +: {ref} better / =: no sig. diff / -: worse ]")
    summ = []
    for k in algos:
        if k == ref:
            continue
        w = t = l = 0
        for fn in funcs:
            x, y = raw[fn, ref], raw[fn, k]
            p = ranksums(x, y).pvalue if (x.std() > 0 or y.std() > 0) else 1.0
            if not np.isfinite(p) or p >= 0.05: t += 1
            elif x.mean() < y.mean(): w += 1
            else: l += 1
        summ.append(dict(vs=k, plus=w, equal=t, minus=l))
    print(pd.DataFrame(summ).to_string(index=False))

    np.savez(f"{a.out}/bench23_convergence.npz", **{f"{fn}|{k}": v for (fn, k), v in hist.items()})
    if a.plots:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            for fn in funcs:
                plt.figure()
                for k in algos:
                    plt.plot(hist[fn, k], label=k)
                if all((hist[fn, k] > 0).all() for k in algos): plt.yscale("log")
                plt.title(fn); plt.xlabel("iteration"); plt.ylabel("best fitness"); plt.legend(); plt.grid(alpha=.3)
                plt.savefig(f"{a.out}/conv_{fn}.png", dpi=150, bbox_inches="tight"); plt.close()
        except Exception as e:
            print("plots skipped:", e)


def fn_stat(v, m):
    return float(v.mean()) if m == "Mean" else (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


if __name__ == "__main__":
    main()
