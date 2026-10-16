import random
import numpy as np
import torch


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def dft_codebook(nt, nb):
    """(nt, nb) unit-norm DFT-like codebook, beams uniform in u = sin(theta) in [-1, 1)."""
    n = np.arange(nt)[:, None]
    u = -1 + 2 * np.arange(nb)[None, :] / nb
    return (np.exp(1j * np.pi * n * u) / np.sqrt(nt)).astype(np.complex64)


def steering(nt, u):
    """Unnormalised ULA steering vectors, (nt, len(u)). u = sin(theta)."""
    u = np.atleast_1d(np.asarray(u, float))
    n = np.arange(nt)[:, None]
    return np.exp(1j * np.pi * n * u[None, :])


def add_data_args(p):
    p.add_argument("--source", default="synthetic", choices=["deepmimo", "synthetic"])
    p.add_argument("--scenario", default="asu_campus_3p5", help="DeepMIMO scenario name")
    p.add_argument("--bs_index", type=int, default=0)
    p.add_argument("--nt", type=int, default=32, help="BS antennas (ULA)")
    p.add_argument("--nbeams", type=int, default=32)
    p.add_argument("--T", type=int, default=8, help="observed steps")
    p.add_argument("--hz", type=int, default=1, help="prediction horizon (steps ahead)")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=4000)
    p.add_argument("--step_min", type=float, default=1.0, help="UE step per snapshot, in grid spacings")
    p.add_argument("--step_max", type=float, default=3.0)
    p.add_argument("--block", type=float, default=20.0, help="spatial block size (m) for leakage-free split")
    p.add_argument("--grid_step", type=float, default=1.0, help="synthetic grid spacing (m)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="results")
    return p


class RowLogger:
    """Append-only CSV: every add() is written + fsynced immediately, so a crash loses nothing.
    resume=True keeps the existing file and lets you skip finished rows via has(); otherwise the file restarts."""

    def __init__(self, path, resume=False):
        import os
        self.path = path; self.rows = []; self.header = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            if resume and os.path.getsize(path) > 0:
                import pandas as pd
                self.rows = pd.read_csv(path).to_dict("records"); self.header = list(self.rows[0].keys()) if self.rows else None
            else:
                os.remove(path)

    def has(self, **kw):
        return any(all(r.get(k) == v for k, v in kw.items()) for r in self.rows)

    def add(self, row):
        import csv, os
        if self.header is None:
            self.header = list(row.keys())
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row); f.flush(); os.fsync(f.fileno())
        self.rows.append(row)

    def df(self):
        import pandas as pd
        return pd.DataFrame(self.rows)
