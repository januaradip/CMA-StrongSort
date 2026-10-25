"""
Re-identification models.

All three are OSNet-x1.0 initialised from Market-1501 and fine-tuned on crops of the TRAINING sequences of the same
benchmark (identities = sequence:track id), with the same sampler, optimiser and schedule (Table 2 of the paper):
PK sampling P = 16, K = 4 (batch 64), Adam 3e-4 with cosine decay, 60 epochs, 256 x 128 crops, horizontal flip.

    ce    classification objective (cross-entropy with label smoothing 0.1 over the training identities); the 512-d
          backbone feature is the embedding. Used by DeepSORT, StrongSORT and ablation rows A and D.
    con   contrastive objective only, L = L_con (Eq. 17, margin m = 1.0), 128-d head. Ablation row B.
    hyb   CMA-StrongSORT: L = L_con + lambda L_tri (Eqs. 17-19), m = 1.0, alpha = 0.3, lambda = 1.0, 128-d head with
          BatchNorm, L2-normalised (Eq. 16). Ablation rows C and E and the CMA-StrongSORT rows of every table.
"""
import io, math, os, random, time

import numpy as np

IMG_H, IMG_W = 256, 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MARKET_URL = 'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA'   # osnet_x1_0_market1501.pt (boxmot model zoo)
KINDS = ('ce', 'con', 'hyb')

CFG = dict(EMB=128, EPOCHS=60, LR=3e-4, BATCH=64, P=16, K=4, MARGIN=1.0, ALPHA=0.3, LAMBDA=1.0, SMOOTH=0.1,
           ITERS_PER_EPOCH=None, SEED=42)


def crop_resize(img, box):
    """img: HxWx3 uint8 RGB; box xyxy in pixels -> 256x128x3 uint8."""
    import cv2
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
    return cv2.resize(np.ascontiguousarray(img[y1:y2, x1:x2]), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)


def to_tensor(crops):
    import torch
    x = (np.asarray(crops, np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def build_osnet(weights=None):
    from boxmot.reid.backbones.families.osnet import osnet_x1_0
    import torch
    m = osnet_x1_0(num_classes=751, pretrained=False, loss='triplet')
    if weights:
        sd = torch.load(weights, map_location='cpu', weights_only=False)
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        own = m.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        m.load_state_dict(ok, strict=False)
        frac = len(ok) / max(1, len([k for k in own if not k.startswith('classifier')]))
        if frac < 0.9:
            raise RuntimeError(f'OSNet weights {weights}: only {frac:.0%} of the layers matched')
    return m


def market_weights(cache_dir):
    """Path of osnet_x1_0_market1501.pt, downloaded once to cache_dir."""
    p = os.path.join(cache_dir, 'osnet_x1_0_market1501.pt')
    if not os.path.exists(p) or os.path.getsize(p) < 1e6:
        import gdown
        os.makedirs(cache_dir, exist_ok=True)
        gdown.download(MARKET_URL, p, quiet=False)
        if not os.path.exists(p) or os.path.getsize(p) < 1e6:
            raise RuntimeError(f'Download of the Market-1501 OSNet weights failed; put osnet_x1_0_market1501.pt in {cache_dir}')
    return p


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


def make_model(kind, market_path, n_ids=0):
    torch, nn, F = _torch()

    class ReID(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.backbone = build_osnet(market_path)
            if kind == 'ce':
                self.emb_dim = 512
                self.classifier = nn.Linear(512, max(1, n_ids))
            else:
                self.emb_dim = CFG['EMB']
                self.head = nn.Sequential(nn.Linear(512, CFG['EMB'], bias=False), nn.BatchNorm1d(CFG['EMB']))

        def features(self, x):
            b = self.backbone
            f = b.global_avgpool(b.featuremaps(x)).flatten(1)
            if getattr(b, 'fc', None) is not None:
                f = b.fc(f)
            return f

        def forward(self, x, logits=False):
            f = self.features(x)
            if self.kind == 'ce':
                return self.classifier(f) if logits else F.normalize(f, dim=1)
            return F.normalize(self.head(f), dim=1)                      # Eq. 16

    return ReID()


class Encoder:
    """embed(list of 256x128 RGB crops) -> (N, D) L2-normalised float32."""

    def __init__(self, model, half=True):
        torch, _, _ = _torch()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device).eval()
        self.half = half and self.device == 'cuda'
        self.dim = model.emb_dim

    def embed(self, crops, bs=256):
        torch, _, F = _torch()
        if len(crops) == 0:
            return np.zeros((0, self.dim), np.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(crops), bs):
                x = to_tensor(crops[s:s + bs]).to(self.device)
                with torch.autocast('cuda', dtype=torch.float16, enabled=self.half):
                    f = self.model(x)
                out.append(F.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(out)


def metric_losses(emb, labels, lam):
    """L_con over all pairs of the batch (Eq. 17) + lam * batch-hard triplet loss (Eq. 18)."""
    torch, _, F = _torch()
    D = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    pos, neg = same & ~eye, ~same
    l_con = ((D[pos] ** 2).sum() + (F.relu(CFG['MARGIN'] - D[neg]) ** 2).sum()) / max(1, int(pos.sum() + neg.sum()))
    if lam == 0:
        return l_con
    d_ap = (D * pos).max(1).values
    d_an = (D + 1e6 * (~neg)).min(1).values
    return l_con + lam * F.relu(d_ap - d_an + CFG['ALPHA']).mean()


def pk_batches(labels, n_batches, rng):
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    ids = list(by)
    P = min(CFG['P'], len(ids))
    K = max(CFG['K'], math.ceil(CFG['BATCH'] / P))
    for _ in range(n_batches):
        batch = []
        for l in rng.sample(ids, P):
            pool = by[l]
            batch += rng.choices(pool, k=K) if len(pool) < K else rng.sample(pool, K)
        yield batch


def train(kind, crops_jpeg, labels, market_path, out_path, log=print, heartbeat=None):
    """Fine-tune one ReID model. crops_jpeg: list of JPEG bytes (256x128 RGB); labels: identity per crop.
    Resumable (out_path.partial keeps the state after every epoch)."""
    assert kind in KINDS
    torch, nn, F = _torch()
    from PIL import Image
    rng = random.Random(CFG['SEED'])
    torch.manual_seed(CFG['SEED'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    uniq = {l: i for i, l in enumerate(sorted(set(labels)))}
    y_all = np.array([uniq[l] for l in labels])
    model = make_model(kind, market_path, len(uniq)).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=CFG['LR'])
    n_iter = CFG['ITERS_PER_EPOCH'] or max(1, len(crops_jpeg) // CFG['BATCH'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['EPOCHS'] * n_iter)
    scaler = torch.amp.GradScaler('cuda', enabled=dev == 'cuda')
    lam = 0.0 if kind == 'con' else CFG['LAMBDA']
    ckpt, start = out_path + '.partial', 0
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        start = s['epoch']
        rng.setstate(s['rng'])
        log(f'  ReID {kind}: resuming at epoch {start}')

    def load(i):
        a = np.asarray(Image.open(io.BytesIO(crops_jpeg[i])).convert('RGB'))
        return a[:, ::-1] if rng.random() < 0.5 else a

    for ep in range(start, CFG['EPOCHS']):
        t0, tot = time.time(), 0.0
        for batch in pk_batches(y_all.tolist(), n_iter, rng):
            x = to_tensor([load(i) for i in batch]).to(dev)
            y = torch.as_tensor(y_all[batch], device=dev)
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev == 'cuda'):
                out = model(x, logits=True) if kind == 'ce' else model(x)
            out = out.float()
            loss = F.cross_entropy(out, y, label_smoothing=CFG['SMOOTH']) if kind == 'ce' else metric_losses(out, y, lam)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                        rng=rng.getstate()), ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)
        if heartbeat:
            heartbeat()
        if ep == start or (ep + 1) % 10 == 0:
            log(f'  ReID {kind} epoch {ep + 1}/{CFG["EPOCHS"]}: loss {tot / n_iter:.4f} ({time.time() - t0:.0f} s/epoch)')
    torch.save(dict(model=model.state_dict(), kind=kind, n_ids=len(uniq), cfg=dict(CFG)), out_path + '.tmp')
    os.replace(out_path + '.tmp', out_path)
    os.remove(ckpt)
    return out_path


def load(path, market_path):
    torch, _, _ = _torch()
    s = torch.load(path, map_location='cpu', weights_only=False)
    m = make_model(s['kind'], market_path, s['n_ids'])
    m.load_state_dict(s['model'])
    return m
