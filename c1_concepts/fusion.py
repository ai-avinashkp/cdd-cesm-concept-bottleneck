"""Idea A: breast-level 4-view fusion over the frozen embeddings of a trained run (Mammo-CLIP fold models).

Each breast = up to 4 tokens (LE-CC, LE-MLO, DES-CC, DES-MLO), each a 1408-d embedding from fold k's model
(extract_features.py). For fold k, a small head is trained on the other folds' breasts and tested on fold k.
Heads (all settings fixed a priori, no tuning on test folds):
  attn  projection 1408->128 + view embedding + 2-layer transformer encoder (4 heads, dropout 0.3) + CLS token
  mean  same projection + view embedding, masked mean over views (isolates the effect of attention)
Both output 35 breast-level concepts (BCE, excluded cells masked) + 3-class label (CE). During training each
present view is dropped with p=0.25 (robustness to missing views). 3 head seeds are averaged.
Then: black box = label head; CBM = logistic regression on predicted breast concepts (other folds -> fold k).
Compared with the current per-image averaging of the same run, with a paired patient bootstrap.

Usage: python fusion.py --run b2_mammoclip_640x1024     (CPU, a few minutes)  -> runs/<run>/FUSION.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from compare_runs import CLASSES, TAB, metrics, model_probs

HERE = Path(__file__).resolve().parent
SLOTS = ["LE_CC", "LE_MLO", "DES_CC", "DES_MLO"]
EPOCHS, LR, WD, DIM, VIEW_DROP, HEAD_SEEDS = 120, 1e-3, 1e-2, 128, 0.25, 3
torch.set_num_threads(2)


class Head(nn.Module):
    def __init__(self, din, nconc, attn):
        super().__init__()
        self.attn = attn
        self.proj = nn.Sequential(nn.LayerNorm(din), nn.Dropout(0.3), nn.Linear(din, DIM))
        self.view = nn.Parameter(torch.zeros(len(SLOTS), DIM))
        if attn:
            self.cls = nn.Parameter(torch.zeros(1, 1, DIM))
            layer = nn.TransformerEncoderLayer(DIM, 4, 2 * DIM, dropout=0.3, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
        self.out = nn.Sequential(nn.LayerNorm(DIM), nn.Dropout(0.3), nn.Linear(DIM, nconc + len(CLASSES)))

    def forward(self, x, m):                       # x: B x 4 x D, m: B x 4 (1 = view present)
        h = self.proj(x) + self.view
        if self.attn:
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            pad = torch.cat([torch.zeros(len(m), 1, dtype=torch.bool), m == 0], 1)
            z = self.enc(h, src_key_padding_mask=pad)[:, 0]
        else:
            w = m.unsqueeze(-1)
            z = (h * w).sum(1) / w.sum(1).clamp(min=1)
        return self.out(z)


def breast_tensors(run, br, k):
    feats = np.load(HERE / "runs" / run / f"feats_fold{k}.npy").astype(np.float32)
    index = {n: i for i, n in enumerate(pd.read_csv(HERE / "runs" / run / "feats_index.csv").Image_name)}
    img = pd.read_csv(TAB / "images.csv")
    first = img.groupby(["breast", "slot"]).Image_name.first()
    X = np.zeros((len(br), len(SLOTS), feats.shape[1]), np.float32)
    M = np.zeros((len(br), len(SLOTS)), np.float32)
    for i, b in enumerate(br.index):
        for j, s in enumerate(SLOTS):
            if (b, s) in first.index:
                X[i, j] = feats[index[first[(b, s)]]]
                M[i, j] = 1
    return X, M


def train_predict(X, M, C, CM, y, tr, te, attn, seed):
    torch.manual_seed(seed)
    mu = X[tr][M[tr] == 1].mean(0)
    sd = X[tr][M[tr] == 1].std(0) + 1e-6
    Xn = torch.tensor((X - mu) / sd * M[..., None])
    Mt, Ct, CMt, yt = torch.tensor(M), torch.tensor(C), torch.tensor(CM), torch.tensor(y)
    nc = C.shape[1]
    net = Head(X.shape[2], nc, attn)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    bce, ce = nn.BCEWithLogitsLoss(reduction="none"), nn.CrossEntropyLoss()
    idx = np.where(tr)[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        for b in np.array_split(idx[torch.randperm(len(idx), generator=g).numpy()], max(1, len(idx) // 32)):
            m = Mt[b].clone()
            drop = (torch.rand(m.shape, generator=g) < VIEW_DROP) & (m == 1)
            keep_one = m.sum(1) - drop.sum(1) < 1                      # never drop every view
            drop[keep_one] = False
            m[drop] = 0
            o = net(Xn[b] * m.unsqueeze(-1), m)
            lc = (bce(o[:, :nc], Ct[b]) * CMt[b]).sum() / CMt[b].sum().clamp(min=1)
            loss = lc + ce(o[:, nc:], yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        o = net(Xn[te], Mt[te])
    return torch.sigmoid(o[:, :nc]).numpy(), torch.softmax(o[:, nc:], 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b2_mammoclip_640x1024")
    a = ap.parse_args()
    br = pd.read_csv(TAB / "breasts.csv").merge(pd.read_csv(TAB / "folds.csv"), on="patient").set_index("breast")
    br["y"] = br.label.map({c: i for i, c in enumerate(CLASSES)})
    C = pd.read_csv(HERE / "concepts_breast.csv").set_index("breast").loc[br.index]
    CM = pd.read_csv(HERE / "concept_mask_breast.csv").set_index("breast").loc[br.index]
    cols = list(C.columns)
    y, folds = br.y.to_numpy(), br.fold.to_numpy()
    out = {}
    for attn in (True, False):
        name = "attention fusion" if attn else "mean fusion"
        pc, pl = np.zeros((len(y), len(cols))), np.zeros((len(y), 3))
        for k in range(5):
            X, M = breast_tensors(a.run, br, k)
            tr, te = folds != k, folds == k
            for s in range(HEAD_SEEDS):
                c_, l_ = train_predict(X, M, C[cols].to_numpy(np.float32), CM[cols].to_numpy(np.float32),
                                       y, tr, te, attn, seed=100 * k + s)
                pc[te] += c_ / HEAD_SEEDS
                pl[te] += l_ / HEAD_SEEDS
            print(f"{name}: fold {k} done", flush=True)
        cbm = np.zeros_like(pl)
        for k in range(5):
            tr, te = folds != k, folds == k
            cbm[te] = LogisticRegression(C=1.0, max_iter=5000).fit(pc[tr], y[tr]).predict_proba(pc[te])
        out[f"{name} / black box"], out[f"{name} / concept bottleneck"] = pl, cbm
    base = model_probs(a.run, br)
    out["per-image averaging (current) / black box"] = base[f"{a.run} / black box"]
    out["per-image averaging (current) / concept bottleneck"] = base[f"{a.run} / concept bottleneck"]

    pats = br.patient.to_numpy()
    up = np.unique(pats)
    idx = {p: np.where(pats == p)[0] for p in up}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(2000)]
    L = [f"# Idea A: breast-level 4-view fusion ({a.run}, frozen fold-model embeddings, {HEAD_SEEDS} head seeds)", "",
         "| model | accuracy | macro AUC | B-vs-M AUC |", "|---|---|---|---|"]
    for n, p in out.items():
        L.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in metrics(p, y)) + " |")
    L += ["", "## Paired differences vs per-image averaging (same kind), patient bootstrap x2000", "",
          "| comparison | accuracy | macro AUC | B-vs-M AUC |", "|---|---|---|---|"]
    for fus in ("attention fusion", "mean fusion"):
        for kind in ("black box", "concept bottleneck"):
            A_, B_ = out[f"{fus} / {kind}"], out[f"per-image averaging (current) / {kind}"]
            d = np.array([metrics(A_[i], y[i]) - metrics(B_[i], y[i]) for i in boots])
            est = metrics(A_, y) - metrics(B_, y)
            lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
            L.append(f"| {fus} vs averaging ({kind}) | " + " | ".join(
                f"{est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}] p={(d[:, j] <= 0).mean():.3f}" for j in range(3)) + " |")
    A_, B_ = out["attention fusion / concept bottleneck"], out["attention fusion / black box"]
    d = np.array([metrics(A_[i], y[i]) - metrics(B_[i], y[i]) for i in boots])
    est = metrics(A_, y) - metrics(B_, y)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    L.append("| attention fusion: CBM vs black box | " + " | ".join(
        f"{est[j]:+.3f} [{lo[j]:+.3f}, {hi[j]:+.3f}] p={(d[:, j] <= 0).mean():.3f}" for j in range(3)) + " |")
    text = "\n".join(L)
    (HERE / "runs" / a.run / "FUSION.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
