import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from common import dft_codebook


# ----------------------------------------------------------------------------- channels
def deepmimo_channels(scenario, nt, bs_index=0):
    """Narrowband channels for every valid grid user. API names follow DeepMIMO v4
    (pip install deepmimo). If your installed version differs, adapt only this function."""
    import deepmimo as dm
    dm.download(scenario)
    ds = dm.load(scenario)
    if not hasattr(ds, "rx_pos"):          # multi-BS scenario -> pick one BS
        ds = ds[bs_index]
    p = dm.ChannelParameters()
    p.bs_antenna.shape = np.array([nt, 1])
    p.ue_antenna.shape = np.array([1, 1])
    p.freq_domain = True
    try:
        p.ofdm.selected_subcarriers = np.array([0])
    except Exception:
        pass
    ch = np.asarray(ds.compute_channels(p))                      # (n_ue, n_rx, n_tx, n_sc)
    ch = ch.reshape(ch.shape[0], ch.shape[2], -1)[:, :, 0]       # (n_ue, nt)
    valid = (np.asarray(ds.los) != -1) & (np.linalg.norm(ch, axis=1) > 0)
    pos = np.asarray(ds.rx_pos)[:, :2]
    bs = np.asarray(ds.tx_pos).reshape(-1, 3)[0, :2]
    return dict(H=ch[valid].astype(np.complex64), pos=pos[valid].astype(np.float32),
                bs_pos=bs.astype(np.float32))


def synthetic_channels(nt, step, fc=28e9):
    """Fallback geometric channel (LoS + 2 wall reflections) on a 2-D grid. For testing the
    pipeline only; NOT a substitute for DeepMIMO ray-tracing data in the paper."""
    lam = 3e8 / fc
    xs = np.arange(20, 120 + 1e-9, step); ys = np.arange(-60, 60 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    H = np.zeros((len(pos), nt), np.complex128)
    n = np.arange(nt)[None, :]
    for w in (None, 70.0, -70.0):
        y = pos[:, 1] if w is None else 2 * w - pos[:, 1]
        d = np.hypot(pos[:, 0], y); u = y / d
        g = (lam / (4 * np.pi * d)) * np.exp(-2j * np.pi * d / lam) * (1.0 if w is None else 0.6)
        H += g[:, None] * np.exp(1j * np.pi * n * u[:, None])
    return dict(H=H.astype(np.complex64), pos=pos.astype(np.float32), bs_pos=np.zeros(2, np.float32))


def get_channels(a):
    os.makedirs(a.cache, exist_ok=True)
    key = f"{a.source}_{a.scenario if a.source == 'deepmimo' else a.grid_step}_{a.nt}"
    path = os.path.join(a.cache, f"channels_{key}.npz")
    if os.path.exists(path):
        z = np.load(path); return {k: z[k] for k in z.files}
    ch = deepmimo_channels(a.scenario, a.nt, a.bs_index) if a.source == "deepmimo" \
        else synthetic_channels(a.nt, a.grid_step)
    np.savez_compressed(path, **ch)
    return ch


# ----------------------------------------------------------------------------- features / splits / trajectories
def make_features(H):
    """Phase-reference to antenna 0, unit-normalise, append standardised log-norm. (n, 2*nt+1)"""
    ref = H * np.exp(-1j * np.angle(H[:, :1]))
    nrm = np.linalg.norm(ref, axis=1, keepdims=True) + 1e-30
    u = ref / nrm
    lg = np.log10(nrm[:, 0]); lg = (lg - lg.mean()) / (lg.std() + 1e-9)
    return np.concatenate([u.real, u.imag, lg[:, None]], 1).astype(np.float32)


def make_splits(pos, seed, block=20.0, ratios=(0.7, 0.1, 0.2)):
    """Spatial-block split: train/val/test users never share a block -> no trajectory leakage."""
    rng = np.random.default_rng(seed)
    cell = np.stack([np.floor(pos[:, 0] / block), np.floor(pos[:, 1] / block)], 1).astype(int)
    keys, inv = np.unique(cell, axis=0, return_inverse=True); inv = inv.ravel()
    if len(keys) < 10:
        print(f"[warn] only {len(keys)} spatial blocks; reduce --block")
    perm = rng.permutation(len(keys)); cut = (np.cumsum(ratios) * len(keys)).astype(int)
    part = np.zeros(len(keys), int)
    part[perm[cut[0]:cut[1]]] = 1; part[perm[cut[1]:]] = 2
    ids = part[inv]
    return {k: np.where(ids == i)[0] for i, k in enumerate(["train", "val", "test"])}


def make_trajs(pos, users, n, L, step_lo, step_hi, seed):
    """n random-walk UE trajectories of L snapshots, snapped to the nearest valid grid user
    inside `users`. Returns global user indices (n, L)."""
    rng = np.random.default_rng(seed)
    P = pos[users].astype(float); tree = cKDTree(P)
    sp = np.median(tree.query(P, k=2)[0][:, 1]); tol = 1.5 * sp
    cur = rng.integers(0, len(P), n); xy = P[cur].copy()
    hd = rng.uniform(0, 2 * np.pi, n); st = rng.uniform(step_lo, step_hi, n) * sp
    out = np.zeros((n, L), np.int64)
    for t in range(L):
        out[:, t] = cur
        hd = hd + rng.normal(0, 0.15, n)
        new = xy + st[:, None] * np.stack([np.cos(hd), np.sin(hd)], 1)
        d, j = tree.query(new); ok = d <= tol
        cur = np.where(ok, j, cur); xy = np.where(ok[:, None], new, xy)
        hd = np.where(ok, hd, hd + np.pi + rng.normal(0, 0.5, n))
    return users[out]


class SeqData:
    def __init__(s, Hf, G, idx, T, hz):
        s.Hf, s.G, s.T, s.hz = Hf, G, T, hz
        s.dev = Hf.device
        s.idx = torch.as_tensor(idx, device=s.dev)

    def __len__(s):
        return s.idx.shape[0]

    def subset(s, n):
        return SeqData(s.Hf, s.G, s.idx[:n].cpu().numpy(), s.T, s.hz)

    def batch(s, sel):
        ii = s.idx[sel]
        X = s.Hf[ii[:, :s.T]]                       # (B, T, F)
        tgt = ii[:, s.T - 1 + s.hz]
        g = s.G[tgt]                                # beam gains at target time (B, nb)
        return X, g.argmax(1), g

    def target_users(s):
        return s.idx[:, s.T - 1 + s.hz].cpu().numpy()


def build_all(a, dev):
    ch = get_channels(a)
    H = ch["H"].astype(np.complex128)
    cb = dft_codebook(a.nt, a.nbeams)
    G = (np.abs(np.conj(H) @ cb) ** 2).astype(np.float32)
    Hf = torch.as_tensor(make_features(H), device=dev)
    Gt = torch.as_tensor(G, device=dev)
    sp = make_splits(ch["pos"], a.seed, a.block)
    L = a.T + a.hz
    sizes = dict(train=a.n_train, val=a.n_val, test=a.n_test)
    data = {}
    for i, (k, u) in enumerate(sp.items()):
        idx = make_trajs(ch["pos"], u, sizes[k], L, a.step_min, a.step_max, a.seed * 10 + i)
        data[k] = SeqData(Hf, Gt, idx, a.T, a.hz)
    aux = dict(H=ch["H"], pos=ch["pos"], bs_pos=ch["bs_pos"], cb=cb, G=G, n_users=len(H))
    print(f"[data] users={len(H)}  train/val/test sequences = "
          f"{len(data['train'])}/{len(data['val'])}/{len(data['test'])}  features={Hf.shape[1]}")
    return data, aux
