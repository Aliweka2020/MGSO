"""Explanation figures + fail-case cause analysis for (a) the optimisers and (b) the ML models.
Every function writes PNG (and CSV where useful) into `out`. torch is imported lazily, so the
optimiser part runs without it."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = plt.get_cmap("tab10")
FAIL_THR = 0.05     # a run "fails" if it closes < 95 % of the gap between start and best-known value


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=180, bbox_inches="tight"); plt.close(fig)
    print("  figure:", os.path.join(out, name))


def _color(names):
    return {n: ("crimson" if n == "MGSO" else COL(i % 10)) for i, n in enumerate(names)}


# =============================================================================== optimisers
def trace_run(cls, f, lb, ub, pop, iters, seed):
    """Same loop as Base.run() but records best-so-far, population diversity and last-improvement iter."""
    o = cls(f, lb, ub, pop=pop, iters=iters, seed=seed)
    o.X = o.lb + o.rng.random((o.pop, o.dim)) * (o.ub - o.lb)
    o.F = o.evaluate(o.X); o.setup()
    span = np.maximum(o.ub - o.lb, 1e-12)
    div = lambda: float(np.mean(np.std(o.X / span, axis=0)))
    hist, dv, last = [o.best_f], [div()], 0
    for t in range(iters):
        prev = o.best_f; o.step(t); hist.append(o.best_f); dv.append(div())
        if o.best_f < prev - 1e-12 * (1 + abs(prev)):
            last = t + 1
    return dict(hist=np.array(hist), div=np.array(dv), last=last, final=o.best_f, x=o.best_x,
                modes=np.array(getattr(o, "mode_log", [])))


def shifted_benchmarks(D, seed=99):
    """Shifted (so centre-biased optimisers get no free lunch) classic functions."""
    r = np.random.default_rng(seed); sh = r.uniform(-.4, .4, D)
    B = {"sphere": 100, "rastrigin": 5.12, "ackley": 32, "rosenbrock": 30, "schwefel": 500}

    def make(fn):
        b = B[fn]

        def f(X):
            Z = X - sh * b
            if fn == "sphere":
                return (Z ** 2).sum(1)
            if fn == "rastrigin":
                return 10 * D + (Z ** 2 - 10 * np.cos(2 * np.pi * Z)).sum(1)
            if fn == "ackley":
                return -20 * np.exp(-.2 * np.sqrt((Z ** 2).mean(1))) - np.exp(np.cos(2 * np.pi * Z).mean(1)) + 20 + np.e
            if fn == "rosenbrock":
                return (100 * (Z[:, 1:] - Z[:, :-1] ** 2) ** 2 + (1 - Z[:, :-1]) ** 2).sum(1)
            return 418.9829 * D - (X * np.sin(np.sqrt(np.abs(X)))).sum(1)     # schwefel (deceptive)
        return f, -b * np.ones(D), b * np.ones(D)
    return {fn: make(fn) for fn in B}


def beam_problems(nt=32, n_inst=3, seed=0):
    """Self-contained multipath ULA channels -> the two beam problems of exp3 (min-form)."""
    r = np.random.default_rng(seed); out = {}
    n = np.arange(nt)[:, None]
    for i in range(n_inst):
        u = r.uniform(-1, 1, 3); g = (r.normal(size=3) + 1j * r.normal(size=3)) * np.array([1, .5, .3])
        h = (np.exp(1j * np.pi * n * u[None]) @ g).ravel(); h /= np.linalg.norm(h); hc = np.conj(h)
        steer = lambda x: np.exp(1j * np.pi * n * np.atleast_1d(x)[None])
        fa = lambda X, hc=hc: -np.abs(hc @ (steer(X[:, 0]) / np.sqrt(nt))) ** 2
        fp = lambda X, hc=hc: -np.abs(np.exp(1j * X) @ hc / np.sqrt(nt)) ** 2
        out[f"beam-angle#{i}"] = (fa, np.array([-1.0]), np.array([1.0]))
        out[f"beam-phase#{i}"] = (fp, np.full(nt, -np.pi), np.full(nt, np.pi))
    return out


def classify_failures(runs, iters):
    """runs: list of dict(opt, prob, hist, div, last, final). Adds gap + fail + cause."""
    ref = {}
    for r in runs:
        ref[r["prob"]] = min(ref.get(r["prob"], np.inf), r["final"])
    for r in runs:
        h0 = r["hist"][0]; best = ref[r["prob"]]
        r["gap"] = (r["final"] - best) / (abs(h0 - best) + 1e-30)
        r["fail"] = r["gap"] > FAIL_THR
        d = r["div"]; mid = d[len(d) // 2] / (d[0] + 1e-30)
        if not r["fail"]:
            r["cause"] = "success"
        elif mid < 0.05:
            r["cause"] = "premature convergence"
        elif r["last"] < 0.6 * iters:
            r["cause"] = "stagnation (diverse but stuck)"
        elif r["last"] > 0.9 * iters:
            r["cause"] = "too slow (still improving at end)"
        else:
            r["cause"] = "trapped in local optimum"
    return runs


CAUSE_ORDER = ["success", "premature convergence", "stagnation (diverse but stuck)",
               "trapped in local optimum", "too slow (still improving at end)"]
CAUSE_COL = ["#4daf4a", "#e41a1c", "#ff7f00", "#984ea3", "#377eb8"]


def optimizer_report(optimizers, problems, out, pop=30, iters=100, runs=5, tag="opt"):
    """problems: dict name -> (f, lb, ub).  optimizers: dict name -> class."""
    names = list(optimizers); cl = _color(names); R = []
    for pn, (f, lb, ub) in problems.items():
        for on, cls in optimizers.items():
            for s in range(runs):
                t = trace_run(cls, f, lb, ub, pop, iters, 1000 + s)
                R.append(dict(opt=on, prob=pn, seed=s, **t))
    classify_failures(R, iters)
    df = pd.DataFrame([{k: r[k] for k in ("opt", "prob", "seed", "final", "gap", "fail", "cause", "last")} for r in R])
    df.to_csv(f"{out}/{tag}_runs.csv", index=False)
    probs = list(problems)

    # 1 convergence
    nc = min(4, len(probs)); nr = int(np.ceil(len(probs) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for a_, pn in zip(ax.ravel(), probs):
        for on in names:
            H = np.array([r["hist"] for r in R if r["opt"] == on and r["prob"] == pn])
            m = np.median(H, 0); a_.plot(m, color=cl[on], lw=2.5 if on == "MGSO" else 1.2, label=on)
            a_.fill_between(range(len(m)), np.percentile(H, 25, 0), np.percentile(H, 75, 0), color=cl[on], alpha=.12)
        ref = min(r["final"] for r in R if r["prob"] == pn); h0 = np.median([r["hist"][0] for r in R if r["prob"] == pn])
        a_.set_title(pn, fontsize=9); a_.set_xlabel("iteration"); a_.grid(alpha=.3)
        if h0 - ref > 0 and ref > 0 and h0 / ref > 50:
            a_.set_yscale("log")
    ax.ravel()[0].legend(fontsize=7); ax.ravel()[0].set_ylabel("best objective (median, IQR)")
    for a_ in ax.ravel()[len(probs):]:
        a_.axis("off")
    _save(fig, out, f"{tag}_1_convergence.png")

    # 2 diversity
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False)
    for a_, pn in zip(ax.ravel(), probs):
        for on in names:
            D = np.array([r["div"] for r in R if r["opt"] == on and r["prob"] == pn])
            a_.plot(np.median(D, 0) / np.median(D[:, 0]), color=cl[on], lw=2.5 if on == "MGSO" else 1.2, label=on)
        a_.set_yscale("log"); a_.set_title(pn, fontsize=9); a_.grid(alpha=.3); a_.set_xlabel("iteration")
    ax.ravel()[0].set_ylabel("population diversity (rel. to start)"); ax.ravel()[0].legend(fontsize=7)
    for a_ in ax.ravel()[len(probs):]:
        a_.axis("off")
    fig.suptitle("Diversity collapsing early = premature convergence", fontsize=10)
    _save(fig, out, f"{tag}_2_diversity.png")

    # 3 gap heatmap + fail rate
    G = df.groupby(["opt", "prob"])["gap"].median().unstack().reindex(index=names, columns=probs)
    Fr = df.groupby(["opt", "prob"])["fail"].mean().unstack().reindex(index=names, columns=probs)
    fig, ax = plt.subplots(1, 2, figsize=(1.0 * len(probs) + 6, 0.5 * len(names) + 2.5))
    for a_, M, ttl, cm in ((ax[0], np.log10(G.values + 1e-6), "log10 median relative gap (lower = better)", "viridis_r"),
                           (ax[1], Fr.values, "fail rate (gap > 5 %)", "Reds")):
        im = a_.imshow(M, cmap=cm, aspect="auto"); a_.set_xticks(range(len(probs))); a_.set_xticklabels(probs, rotation=60, ha="right", fontsize=7)
        a_.set_yticks(range(len(names))); a_.set_yticklabels(names); a_.set_title(ttl, fontsize=9); plt.colorbar(im, ax=a_)
    _save(fig, out, f"{tag}_3_gap_failrate.png")

    # 4 fail causes
    C = df.groupby(["opt", "cause"]).size().unstack(fill_value=0).reindex(index=names, columns=CAUSE_ORDER, fill_value=0)
    C = C.div(C.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 3, 4)); bot = np.zeros(len(names))
    for c, col in zip(CAUSE_ORDER, CAUSE_COL):
        ax.bar(names, C[c].values, bottom=bot, color=col, label=c); bot += C[c].values
    ax.set_ylabel("fraction of runs"); ax.legend(fontsize=7, bbox_to_anchor=(1, 1)); ax.set_title("Why do runs fail? (cause classification)")
    _save(fig, out, f"{tag}_4_fail_causes.png"); C.to_csv(f"{out}/{tag}_fail_causes.csv")

    # 5 operator usage for MGSO
    ms = [r for r in R if r["opt"] == "MGSO" and len(r["modes"])]
    if ms:
        M = np.mean([r["modes"] for r in ms], 0); M = M / M.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.stackplot(range(len(M)), M.T, labels=["undulate (explore)", "glide (DE-pbest, adaptive)", "dive (relaunch stagnant)", "landing (local search)"],
                     colors=["#80b1d3", "#8dd3c7", "#fb8072", "#fdb462"])
        ax.set_xlabel("iteration"); ax.set_ylabel("share of population"); ax.legend(fontsize=7, loc="upper right"); ax.set_title("MGSO operator usage")
        _save(fig, out, f"{tag}_5_mgso_operators.png")

    # 6 final-value boxplots (ranks)
    rk = df.groupby(["prob", "seed", "opt"])["final"].mean().unstack().rank(axis=1)
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 2, 3.5))
    ax.boxplot([rk[n].values for n in names], labels=names, patch_artist=True,
               boxprops=dict(facecolor="#dddddd"), medianprops=dict(color="crimson"))
    ax.set_ylabel("rank over (problem, seed)  [1 = best]"); ax.grid(alpha=.3, axis="y")
    _save(fig, out, f"{tag}_6_rank_distribution.png")
    print("  mean rank:", rk.mean().round(2).to_dict())
    return df


def plot_final_box(res, out, name):
    """res: dict opt -> (instances, runs) normalised-gain array (exp3)."""
    fig, ax = plt.subplots(figsize=(7, 3.5)); names = list(res); cl = _color(names)
    bp = ax.boxplot([res[k].ravel() for k in names], labels=names, patch_artist=True)
    for p, k in zip(bp["boxes"], names):
        p.set_facecolor(cl[k]); p.set_alpha(.5)
    ax.set_ylabel("normalised gain (1 = optimum)"); ax.grid(alpha=.3, axis="y")
    ax.set_title("Spread over instances x runs (long lower whiskers = failed runs)")
    _save(fig, out, name)


def plot_exp4(df, out):
    """df: per-user rows of exp4. Optimizer-failure vs link-budget-failure analysis."""
    meths = [m for m in df["method"].unique()]; cl = _color(meths)
    orc = df[df.method.str.startswith("oracle")].set_index("user")["ok"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].boxplot([df[df.method == m]["ee"] for m in meths], labels=meths, patch_artist=True)
    ax[0].set_title("energy efficiency"); ax[0].tick_params(axis="x", rotation=60)
    fr = [df[df.method == m]["ok"].mean() for m in meths]
    ax[1].bar(meths, fr, color=[cl[m] for m in meths]); ax[1].set_title("feasible fraction (rate >= Rmin)"); ax[1].tick_params(axis="x", rotation=60)
    for m in meths:
        d = df[df.method == m]; ax[2].scatter(d["p"], d["rate"], s=8, alpha=.5, color=cl[m], label=m)
    ax[2].set_xlabel("transmit power"); ax[2].set_ylabel("rate"); ax[2].legend(fontsize=6); ax[2].set_title("rate-power operating points")
    _save(fig, out, "exp4_1_overview.png")
    rows = []
    for m in meths:
        d = df[df.method == m]; bad = d[~d.ok]
        link_lim = int((~orc.reindex(bad.user).fillna(False).astype(bool)).sum())
        rows.append(dict(method=m, ok=int(d.ok.sum()), fail_link_budget=link_lim, fail_optimizer=len(bad) - link_lim))
    c = pd.DataFrame(rows).set_index("method"); c.to_csv(f"{out}/exp4_fail_causes.csv")
    fig, ax = plt.subplots(figsize=(8, 3.8)); bot = np.zeros(len(c))
    for col, k, lab in (("#4daf4a", "ok", "feasible"), ("#984ea3", "fail_link_budget", "fail: even oracle infeasible (candidate set / link budget)"),
                        ("#e41a1c", "fail_optimizer", "fail: oracle feasible -> optimiser/baseline fault")):
        ax.bar(c.index, c[k], bottom=bot, color=col, label=lab); bot += c[k].values
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=60); ax.set_ylabel("users"); ax.set_title("Why users miss Rmin")
    _save(fig, out, "exp4_2_fail_causes.png")


# =============================================================================== ML
def plot_training_curves(hists, out):
    if not hists:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6)); cl = _color(list(hists))
    for n, h in hists.items():
        if not h:
            continue
        e = np.arange(1, len(h) + 1)
        ax[0].plot(e, [x["train_loss"] for x in h], color=cl[n], label=n)
        ax[1].plot(e, [x["val_loss"] for x in h], color=cl[n]); ax[2].plot(e, [x["val_top1"] for x in h], color=cl[n])
    for a_, t in zip(ax, ("train loss", "val loss  (rising while train falls = overfitting)", "val top-1")):
        a_.set_title(t, fontsize=9); a_.set_xlabel("epoch"); a_.grid(alpha=.3)
    ax[0].legend(fontsize=7); _save(fig, out, "ml_1_training_curves.png")


def plot_tuner_convergence(tuned, out):
    """tuned: name -> best-so-far validation loss history of the metaheuristic tuner."""
    if not tuned:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for n, h in tuned.items():
        ax.plot(h, marker="o", ms=3, label=n)
    ax.set_xlabel("tuner iteration"); ax.set_ylabel("best val loss"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Hyper-parameter search progress")
    _save(fig, out, "ml_0_tuner_convergence.png")


def ml_report(models, te, aux, out, hists=None):
    """models: name -> trained torch model.  te: SeqData(test).  Explains accuracy AND failures."""
    import torch
    from models import predict_logits
    os.makedirs(out, exist_ok=True)
    names = list(models); cl = _color(names)
    tgt = te.target_users(); G = te.G[torch.as_tensor(tgt, device=te.dev)].cpu().numpy()
    y = G.argmax(1); nb = G.shape[1]
    lognorm = te.Hf[torch.as_tensor(tgt, device=te.dev), -1].cpu().numpy()
    weak = lognorm < np.quantile(lognorm, .25)
    dist = np.linalg.norm(aux["pos"][tgt] - np.asarray(aux["bs_pos"])[None], axis=1)
    P = {}
    for n, m in models.items():
        lg = predict_logits(m, te).cpu(); pr = torch.softmax(lg, 1).numpy()
        P[n] = dict(pred=pr.argmax(1), conf=pr.max(1), prob=pr)
    plot_training_curves(hists, out)

    # 2 confusion matrix of best model + per-beam accuracy
    best = max(names, key=lambda n: (P[n]["pred"] == y).mean())
    cm = np.zeros((nb, nb)); np.add.at(cm, (y, P[best]["pred"]), 1); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    im = ax[0].imshow(cmn, cmap="magma"); plt.colorbar(im, ax=ax[0]); ax[0].set_xlabel("predicted beam"); ax[0].set_ylabel("true beam")
    ax[0].set_title(f"confusion (row-normalised) - {best}\nmass near diagonal = adjacent-beam confusion")
    for n in names:
        acc = [(P[n]["pred"][y == b] == b).mean() if (y == b).any() else np.nan for b in range(nb)]
        ax[1].plot(acc, color=cl[n], label=n, marker=".")
    ax[1].set_xlabel("true beam index"); ax[1].set_ylabel("top-1 accuracy"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    ax[1].set_title("accuracy per beam (dips = rarely-seen or ambiguous beams)")
    _save(fig, out, "ml_2_confusion_per_beam.png")

    # 3 beam-index error histogram + gain-ratio loss CDF
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for n in names:
        d = np.abs(P[n]["pred"] - y); d = d[d > 0]
        ax[0].hist(np.minimum(d, 10), bins=np.arange(.5, 11.5), histtype="step", lw=2 if n == "MGSO-Transformer" else 1.3, color=cl[n], label=n)
        gr = G[np.arange(len(y)), P[n]["pred"]] / np.maximum(G.max(1), 1e-30)
        loss_db = -10 * np.log10(np.clip(gr, 1e-6, 1)); s = np.sort(loss_db)
        ax[1].plot(s, np.linspace(0, 1, len(s)), color=cl[n], label=n)
    ax[0].set_xlabel("|predicted - true| beam index (10 = >=10)"); ax[0].set_ylabel("count (errors only)"); ax[0].legend(fontsize=7)
    ax[0].set_title("how wrong are the mistakes?")
    ax[1].set_xlim(0, 10); ax[1].set_xlabel("beamforming gain loss vs best beam (dB)"); ax[1].set_ylabel("CDF"); ax[1].grid(alpha=.3)
    ax[1].set_title("what do mistakes cost? (steep early = mostly harmless)")
    _save(fig, out, "ml_3_error_size_and_cost.png")

    # 4 accuracy vs distance and vs channel strength
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for xs, a_, lab in ((dist, ax[0], "distance to BS (m)"), (lognorm, ax[1], "standardised log channel norm (SNR proxy)")):
        qs = np.quantile(xs, np.linspace(0, 1, 9)); idx = np.clip(np.digitize(xs, qs[1:-1]), 0, 7); ctr = [xs[idx == i].mean() for i in range(8)]
        for n in names:
            a_.plot(ctr, [(P[n]["pred"][idx == i] == y[idx == i]).mean() for i in range(8)], color=cl[n], label=n, marker="o", ms=3)
        a_.set_xlabel(lab); a_.set_ylabel("top-1 accuracy"); a_.grid(alpha=.3)
    ax[0].legend(fontsize=7); ax[0].set_title("failure vs range"); ax[1].set_title("failure vs signal strength")
    _save(fig, out, "ml_4_accuracy_vs_range_and_snr.png")

    # 5 calibration
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); ax.plot([0, 1], [0, 1], "k--", lw=1)
    for n in names:
        b = np.clip((P[n]["conf"] * 10).astype(int), 0, 9)
        pts = [(P[n]["conf"][b == i].mean(), (P[n]["pred"][b == i] == y[b == i]).mean()) for i in range(10) if (b == i).sum() > 20]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=3, color=cl[n], label=n)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); ax.set_title("calibration (below diagonal = over-confident)")
    _save(fig, out, "ml_5_calibration.png")

    # 6 failure-cause classification of every wrong top-1 prediction
    causes = ["near-tie (<=1 dB loss, harmless)", "adjacent beam (|d|<=1)", "weak channel (lowest-SNR quartile)",
              "far beam (|d|>3)", "other moderate error"]
    ccol = ["#a6d854", "#66c2a5", "#8da0cb", "#e41a1c", "#ffd92f"]; tab = {}
    for n in names:
        pr = P[n]["pred"]; wrong = pr != y; d = np.abs(pr - y)
        gr = G[np.arange(len(y)), pr] / np.maximum(G.max(1), 1e-30)
        c = np.full(len(y), 4)
        c[wrong & (d > 3)] = 3; c[wrong & weak] = 2; c[wrong & (d <= 1)] = 1; c[wrong & (gr >= 0.794)] = 0
        tab[n] = [(wrong & (c == k)).sum() / len(y) for k in range(5)] + [1 - wrong.mean()]
    T = pd.DataFrame(tab, index=causes + ["correct"]).T; T.to_csv(f"{out}/ml_fail_causes.csv")
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 4.2)); bot = np.zeros(len(names))
    for k, (c, col) in enumerate(zip(causes + ["correct"], ccol + ["#dddddd"])):
        ax.bar(names, T[c].values, bottom=bot, color=col, label=c); bot += T[c].values
    ax.set_ylabel("fraction of test sequences"); ax.legend(fontsize=7, bbox_to_anchor=(1, 1)); ax.tick_params(axis="x", rotation=30)
    ax.set_title("Why do beam predictions fail? (priority: near-tie > adjacent > weak channel > far)")
    _save(fig, out, "ml_6_fail_causes.png")
    return T
