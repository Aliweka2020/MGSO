import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(s, in_dim, n_out, ch=64, dropout=0.1, **_):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv1d(in_dim, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, 2 * ch, 3, padding=1), nn.BatchNorm1d(2 * ch), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        s.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * ch, n_out))

    def forward(s, x):                       # x: (B, T, F)
        return s.head(s.net(x.transpose(1, 2)).squeeze(-1))


class RNNNet(nn.Module):
    def __init__(s, in_dim, n_out, kind="lstm", hidden=128, layers=2, dropout=0.1, **_):
        super().__init__()
        s.bi = kind == "bilstm"; s.hid = hidden
        cell = nn.GRU if kind == "gru" else nn.LSTM
        s.rnn = cell(in_dim, hidden, layers, batch_first=True,
                     dropout=dropout if layers > 1 else 0.0, bidirectional=s.bi)
        s.head = nn.Linear(hidden * (2 if s.bi else 1), n_out)

    def forward(s, x):
        o, _ = s.rnn(x)
        h = torch.cat([o[:, -1, :s.hid], o[:, 0, s.hid:]], 1) if s.bi else o[:, -1]
        return s.head(h)


class TransformerNet(nn.Module):
    def __init__(s, in_dim, n_out, d_model=64, nhead=4, layers=2, ff=4, dropout=0.1,
                 max_len=64, pool="last", **_):
        super().__init__()
        s.pool = pool
        s.inp = nn.Linear(in_dim, d_model)
        s.pos = nn.Parameter(torch.zeros(1, max_len, d_model)); nn.init.trunc_normal_(s.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * ff, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        s.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        s.norm = nn.LayerNorm(d_model); s.head = nn.Linear(d_model, n_out)

    def forward(s, x):
        h = s.norm(s.enc(s.inp(x) + s.pos[:, :x.size(1)]))
        return s.head(h[:, -1] if s.pool == "last" else h.mean(1))


def build_model(name, in_dim, n_out, max_len, **hp):
    n = name.lower()
    if n == "cnn":
        return CNN1D(in_dim, n_out, **hp)
    if n in ("lstm", "bilstm", "gru"):
        return RNNNet(in_dim, n_out, kind=n, **hp)
    if n == "transformer":
        return TransformerNet(in_dim, n_out, max_len=max_len, **hp)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def evaluate(model, data, bs=2048):
    model.eval(); n = len(data); L = t1 = t3 = 0.0; gr = 0.0
    for i in range(0, n, bs):
        sel = torch.arange(i, min(i + bs, n), device=data.dev)
        X, y, g = data.batch(sel); lo = model(X)
        L += F.cross_entropy(lo, y, reduction="sum").item()
        top = lo.topk(min(3, lo.size(1)), 1).indices
        t1 += (top[:, 0] == y).sum().item(); t3 += (top == y[:, None]).any(1).sum().item()
        gr += (g.gather(1, top[:, :1]).squeeze(1) / g.max(1).values.clamp_min(1e-30)).sum().item()
    return dict(loss=L / n, top1=t1 / n, top3=t3 / n, gain_ratio=gr / n)


@torch.no_grad()
def predict_logits(model, data, bs=2048):
    model.eval(); out = []
    for i in range(0, len(data), bs):
        sel = torch.arange(i, min(i + bs, len(data)), device=data.dev)
        out.append(model(data.batch(sel)[0]))
    return torch.cat(out)


def fit_eval(model, tr, va, te=None, epochs=30, bs=256, lr=1e-3, wd=1e-2, patience=6):
    dev = tr.dev; model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best, bad, state, vm = 1e9, 0, None, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(tr), device=dev)
        for i in range(0, len(tr), bs):
            X, y, _ = tr.batch(perm[i:i + bs]); loss = F.cross_entropy(model(X), y)
            if not torch.isfinite(loss):
                return dict(val=dict(loss=1e3, top1=0, top3=0, gain_ratio=0), test=None, params=n_params(model))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        vm = evaluate(model, va)
        if vm["loss"] < best - 1e-4:
            best, bad, state = vm["loss"], 0, {k: v.clone() for k, v in model.state_dict().items()}
            bvm = vm
        else:
            bad += 1
            if bad >= patience:
                break
    if state is not None:
        model.load_state_dict(state)
    return dict(val=bvm, test=evaluate(model, te) if te is not None else None,
                params=n_params(model), epochs_run=ep + 1)


def save_ckpt(path, model, name, hp, in_dim, n_out, max_len):
    torch.save(dict(state=model.state_dict(), name=name, hp=hp, in_dim=in_dim,
                    n_out=n_out, max_len=max_len), path)


def load_ckpt(path, dev):
    c = torch.load(path, map_location=dev)
    m = build_model(c["name"], c["in_dim"], c["n_out"], c["max_len"], **c["hp"])
    m.load_state_dict(c["state"]); return m.to(dev).eval()
