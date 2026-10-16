"""Ablation study for MGSO on the beam-optimisation problems of exp3.

python exp3_ablation.py --source deepmimo --scenario asu_campus_3p5 --problem phase --instances 10 --runs 5 --study loo
studies: loo (leave-one-out) | inc (incremental from core) | only (core + one component) | all
Same seeds (1000*i + r) and same budget for every variant -> paired comparison.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import add_data_args
from data import get_channels
from exp3_beam_optimization import make_problem
from optimizers import get_optimizers
from mgso_ablation import make_variant, LEAVE_ONE_OUT, INCREMENTAL, ONLY_ONE


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--problem", default="phase", choices=["angle", "phase"])
    ap.add_argument("--study", default="loo", choices=["loo", "inc", "only", "all"])
    ap.add_argument("--baselines", default="GSO,DE,GA", help="reference optimizers from optimizers.py ('' for none)")
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    studies = {"loo": LEAVE_ONE_OUT, "inc": INCREMENTAL, "only": ONLY_ONE}
    chosen = list(studies) if a.study == "all" else [a.study]
    algos = {}                                   # name -> class
    for st in chosen:
        for name, off in studies[st].items():
            key = name if st == "loo" else f"[{st}] {name}"
            algos[key] = make_variant(key, **off)
    if a.baselines:
        algos.update(get_optimizers(a.baselines.split(",")))
    full_key = next(k for k in algos if k.endswith("MGSO-full") or k == "[inc] +undulate(=full)")

    ch = get_channels(a); rng = np.random.default_rng(a.seed)
    users = rng.choice(len(ch["H"]), a.instances, replace=False)
    res = {k: np.zeros((a.instances, a.runs)) for k in algos}
    usage = {k: [] for k in algos if hasattr(algos[k], "flags")}     # mode usage fractions (und, glide, dive, land)

    for i, u in enumerate(users):
        h = ch["H"][u].astype(np.complex128); h /= np.linalg.norm(h)
        f, lb, ub, ref = make_problem(h, a.problem)
        for k, cls in algos.items():
            for r in range(a.runs):
                opt = cls(f, lb, ub, pop=a.pop, iters=a.iters, seed=1000 * i + r)
                x, fv, hh = opt.run()
                res[k][i, r] = -fv / ref
                if k in usage and getattr(opt, "mode_log", None):
                    m = np.array(opt.mode_log, float); usage[k].append(m.sum(0) / m.sum())
        print(f"instance {i + 1}/{a.instances} done", flush=True)

    inst = {k: v.mean(1) for k, v in res.items()}                    # per-instance means
    rows = []
    for k in algos:
        try:
            p = wilcoxon(inst[full_key], inst[k]).pvalue if k != full_key else np.nan
        except ValueError:
            p = np.nan
        u = np.mean(usage[k], 0) if k in usage and usage[k] else [np.nan] * 4
        rows.append(dict(variant=k, mean=res[k].mean(), std_inst=inst[k].std(), worst_run=res[k].min(),
                         delta_vs_full=res[k].mean() - res[full_key].mean(), p_vs_full=p,
                         und=u[0], glide=u[1], dive=u[2], land=u[3]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{a.out}/exp3_ablation_{a.problem}_{a.study}.csv", index=False)
    print(f"\nAblation ({a.problem}), budget={a.pop * (a.iters + 1)} evals/run, "
          f"{a.instances} instances x {a.runs} runs. delta/p are relative to '{full_key}'")
    print(df.round(4).to_string(index=False))
    print("\nNote: p-values are NOT corrected for multiple comparisons (use Holm/Bonferroni if you report them).")


if __name__ == "__main__":
    main()
