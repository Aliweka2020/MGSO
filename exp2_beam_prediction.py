"""Experiment 2 - beam prediction: CNN / LSTM / BiLSTM / GRU / Transformer / <Tuner>-Transformer.
<Tuner>-Transformer = Transformer whose hyper-parameters are searched by a metaheuristic
(GSO by default; any other optimizer via --tuners PSO,GA,...)."""
import argparse, json, os
import numpy as np, pandas as pd, torch
from common import add_data_args, set_seed, device, RowLogger
from data import build_all
from models import build_model, fit_eval, save_ckpt
from optimizers import get_optimizers

HP = dict(cnn=dict(ch=64, dropout=0.1),
          lstm=dict(hidden=128, layers=2, dropout=0.1),
          bilstm=dict(hidden=96, layers=2, dropout=0.1),
          gru=dict(hidden=128, layers=2, dropout=0.1),
          transformer=dict(d_model=64, nhead=4, layers=2, ff=4, dropout=0.1))
LR = 1e-3


def pick(lst, v):
    return lst[min(int(v * len(lst)), len(lst) - 1)]


def decode_tf(x):
    """x in [0,1]^6 -> Transformer hyper-parameters + learning rate."""
    hp = dict(d_model=pick([32, 64, 128], x[0]), nhead=pick([1, 2, 4, 8], x[1]),
              layers=pick([1, 2, 3, 4], x[2]), ff=pick([2, 4], x[3]), dropout=0.3 * float(x[4]))
    return hp, 10 ** (-4 + 1.7 * float(x[5]))


def tune(opt_cls, tr, va, in_dim, n_out, a):
    sub = tr.subset(a.tune_n); cache = {}

    def f(X):
        out = []
        for x in X:
            key = tuple(np.round(x, 3))
            if key not in cache:
                hp, lr = decode_tf(x); set_seed(a.seed)
                m = build_model("transformer", in_dim, n_out, a.T, **hp)
                r = fit_eval(m, sub, va, None, epochs=a.tune_epochs, lr=lr, patience=99)
                cache[key] = r["val"]["loss"]
            out.append(cache[key])
        return np.array(out)

    o = opt_cls(f, np.zeros(6), np.ones(6), pop=a.tune_pop, iters=a.tune_iters, seed=a.seed)
    x, fv, hist = o.run()
    hp, lr = decode_tf(x)
    return hp, lr, hist


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--models", default="cnn,lstm,bilstm,gru,transformer")
    ap.add_argument("--tuners", default="GSO", help="comma list; '' to skip")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tune_pop", type=int, default=8)
    ap.add_argument("--tune_iters", type=int, default=6)
    ap.add_argument("--tune_epochs", type=int, default=5)
    ap.add_argument("--tune_n", type=int, default=20000)
    ap.add_argument("--resume", action="store_true", help="continue from existing results/exp2_runs.csv")
    a = ap.parse_args(); dev = device()
    os.makedirs(a.out, exist_ok=True); os.makedirs("ckpt", exist_ok=True)
    data, aux = build_all(a, dev); tr, va, te = data["train"], data["val"], data["test"]
    in_dim, n_out = tr.Hf.shape[1], a.nbeams

    runs = [(m, HP[m], LR) for m in a.models.split(",") if m]
    for t, cls in get_optimizers([x for x in a.tuners.split(",") if x]).items():
        jp = f"{a.out}/tuned_{t}.json"
        if a.resume and os.path.exists(jp):
            j = json.load(open(jp)); hp, lr = j["hp"], j["lr"]
        else:
            hp, lr, hist = tune(cls, tr, va, in_dim, n_out, a)
            json.dump(dict(hp=hp, lr=lr, hist=hist.tolist()), open(jp, "w"))
        print(f"[{t}] best Transformer hp={hp} lr={lr:.2e}")
        runs.append((f"{t}-Transformer", hp, lr))

    log = RowLogger(f"{a.out}/exp2_runs.csv", resume=a.resume)
    for name, hp, lr in runs:
        base = "transformer" if name.endswith("Transformer") else name
        for s in range(a.seeds):
            if log.has(model=name, seed=s):
                continue
            set_seed(a.seed + s)
            m = build_model(base, in_dim, n_out, a.T, **hp)
            r = fit_eval(m, tr, va, te, epochs=a.epochs, lr=lr)
            log.add(dict(model=name, seed=s, params=r["params"], **{f"test_{k}": v for k, v in r["test"].items()}))
            print(name, s, {k: round(v, 4) for k, v in r["test"].items()}, r["params"], "params")
            if s == 0 and base == "transformer":
                save_ckpt(f"ckpt/{name.lower()}.pt", m, base, hp, in_dim, n_out, a.T)
    df = log.df()
    summ = df.groupby("model").agg(["mean", "std"]).drop(columns="seed")
    summ.to_csv(f"{a.out}/exp2_summary.csv"); print(summ.round(4).to_string())
    print("Note: with few seeds use >=10 and a paired test (e.g. Wilcoxon) before claiming significance.")


if __name__ == "__main__":
    main()
