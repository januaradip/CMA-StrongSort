"""
Trackers.

    SORT            Kalman filter (constant velocity, xyah) + IoU matching (Bewley et al., 2016).
    DeepSORT        Matching cascade on appearance (feature gallery, budget 100) gated by the Mahalanobis distance,
                    then IoU matching (Wojke et al., 2017).
    StrongSORT      DeepSORT with the StrongSORT changes (Du et al., 2023): NSA Kalman filter, EMA appearance feature
                    per track, camera motion compensation (ECC), cost C = lambda C_app + (1 - lambda) C_motion with
                    lambda = 0.98 (gamma = 1 - lambda = 0.02 in Eq. 21 of the paper), no matching cascade.
    CMA-StrongSORT  StrongSORT + contrastive ReID embeddings + persistent identity memory (Algorithm 1): before a new
                    identity is created its embedding is compared with the memory bank of the identities that are not
                    currently active (Eq. 23); if cos > tau (Eq. 24) the stored identity is reused, otherwise a new
                    one is written. Matched identities update their memory vector by EMA with beta (Eq. 25).
                    The memory has no size limit and no expiration.
ByteTrack and OC-SORT come from boxmot (BoxmotTracker).

Every tracker takes detections as an (N, 5) array [x1, y1, x2, y2, conf] in pixels and, if it uses appearance, an
(N, D) array of embeddings aligned with the detections; update() returns [(identity, box_xyxy)] for the tracks that
are confirmed and were matched in the current frame.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment

CHI2_4DOF_95 = 9.4877


# =============================================================================== Kalman filter (x, y, a, h)
class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height) as in DeepSORT. With nsa=True the measurement noise
    is scaled by (1 - confidence) (NSA Kalman filter of StrongSORT)."""
    _std_pos, _std_vel = 1. / 20, 1. / 160

    def __init__(self, nsa=False):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.nsa = nsa

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        mean = self.F @ mean
        cov = self.F @ cov @ self.F.T + np.diag(np.square(std))
        return mean, cov

    def project(self, mean, cov, conf=None):
        h = mean[3]
        std = np.array([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h])
        if self.nsa and conf is not None:
            std = std * (1 - conf)
        return self.H @ mean, self.H @ cov @ self.H.T + np.diag(np.square(std))

    def update(self, mean, cov, z, conf=None):
        pm, pc = self.project(mean, cov, conf)
        K = np.linalg.solve(pc, (cov @ self.H.T).T).T
        mean = mean + K @ (z - pm)
        cov = cov - K @ pc @ K.T
        return mean, cov

    def gating_distance(self, mean, cov, zs):
        pm, pc = self.project(mean, cov)
        L = np.linalg.cholesky(pc)
        d = zs - pm
        y = solve_triangular(L, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(y * y, axis=0)


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), max(h, 1e-6)])


def xyah_to_xyxy(m):
    w = m[2] * m[3]
    return np.array([m[0] - w / 2, m[1] - m[3] / 2, m[0] + w / 2, m[1] + m[3] / 2])


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def hungarian(cost, thresh):
    """Minimum-cost assignment (Eq. 15); pairs with cost > thresh are rejected."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(np.isfinite(cost), cost, 1e5)
    r, k = linear_sum_assignment(c)
    matches = [(i, j) for i, j in zip(r, k) if c[i, j] <= thresh]
    mi = {i for i, _ in matches}; mj = {j for _, j in matches}
    return matches, [i for i in range(cost.shape[0]) if i not in mi], [j for j in range(cost.shape[1]) if j not in mj]


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


# =============================================================================== tracks
class Track:
    def __init__(self, tid, mean, cov, conf, feat, n_init, max_age, ema, budget):
        self.id, self.mean, self.cov, self.conf = tid, mean, cov, conf
        self.hits, self.age, self.time_since_update = 1, 1, 0
        self.state = 'confirmed' if n_init <= 1 else 'tentative'
        self.n_init, self.max_age, self.ema, self.budget = n_init, max_age, ema, budget
        self.feat = None if feat is None else _unit(feat)
        self.gallery = [] if feat is None else [self.feat]

    def predict(self, kf):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def apply_warp(self, W):
        """Camera motion compensation: move the predicted box with the 2x3 affine warp of the frame."""
        x1, y1, x2, y2 = self.box
        p1 = W @ np.array([x1, y1, 1.]); p2 = W @ np.array([x2, y2, 1.])
        w, h = p2[0] - p1[0], p2[1] - p1[1]
        if w > 1 and h > 1:
            self.mean[:4] = [p1[0] + w / 2, p1[1] + h / 2, w / h, h]

    def update(self, kf, box, conf, feat):
        self.mean, self.cov = kf.update(self.mean, self.cov, xyxy_to_xyah(box), conf)
        self.conf = conf
        if feat is not None:
            f = _unit(feat)
            if self.ema is None:                                    # DeepSORT: gallery of the last `budget` features
                self.feat = f
                self.gallery = (self.gallery + [f])[-self.budget:]
            else:                                                   # StrongSORT: EMA feature
                self.feat = f if self.feat is None else _unit(self.ema * self.feat + (1 - self.ema) * f)
        self.hits += 1
        self.time_since_update = 0
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        if self.state == 'tentative' or self.time_since_update > self.max_age:
            self.state = 'deleted'

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)


# =============================================================================== SORT / DeepSORT / StrongSORT
class SortLike:
    """appearance=False: SORT. appearance=True, cascade=True, ema=None: DeepSORT. appearance=True, cascade=False,
    ema=0.9, nsa=True: StrongSORT."""

    def __init__(self, appearance=True, min_conf=0.5, mc_lambda=0.98, max_cos_dist=0.2, max_iou_dist=0.7,
                 max_age=30, n_init=3, ema=0.9, nsa=True, cascade=False, budget=100, iou_thresh_sort=0.3):
        self.appearance, self.cascade = appearance, cascade
        self.kf = KalmanXYAH(nsa=nsa and appearance)
        self.min_conf, self.mc_lambda, self.max_cos_dist = min_conf, mc_lambda, max_cos_dist
        self.max_iou_dist = max_iou_dist if appearance else 1 - iou_thresh_sort
        self.max_age, self.n_init, self.ema, self.budget = max_age, n_init, ema, budget
        self.tracks, self.next_id = [], 1

    def _app_cost(self, T, feats):
        if self.ema is None:
            return np.stack([1 - (np.stack(t.gallery) @ feats.T).max(0) for t in T])
        return 1 - np.stack([t.feat for t in T]) @ feats.T          # Eq. 20

    def _appearance_match(self, tidx, boxes, feats, didx):
        if not tidx or not didx:
            return [], tidx, didx
        T = [self.tracks[i] for i in tidx]
        app = self._app_cost(T, feats[didx])
        zs = np.stack([xyxy_to_xyah(boxes[j]) for j in didx])
        cost = np.empty_like(app)
        for r, t in enumerate(T):
            g = self.kf.gating_distance(t.mean, t.cov, zs)
            cost[r] = self.mc_lambda * app[r] + (1 - self.mc_lambda) * g   # Eq. 21
            cost[r, g > CHI2_4DOF_95] = np.inf
        cost[app > self.max_cos_dist] = np.inf
        m, ut, ud = hungarian(cost, self.max_cos_dist)
        return [(tidx[i], didx[j]) for i, j in m], [tidx[i] for i in ut], [didx[j] for j in ud]

    def _cascade_match(self, tidx, boxes, feats, didx):
        matches, left = [], list(didx)
        for level in range(self.max_age):
            if not left:
                break
            lv = [i for i in tidx if self.tracks[i].time_since_update == 1 + level]
            m, _, left = self._appearance_match(lv, boxes, feats, left)
            matches += m
        mt = {i for i, _ in matches}
        return matches, [i for i in tidx if i not in mt], left

    def _iou_match(self, tidx, boxes, didx):
        if not tidx or not didx:
            return [], tidx, didx
        cost = 1 - iou_matrix([self.tracks[i].box for i in tidx], boxes[didx])
        m, ut, ud = hungarian(cost, self.max_iou_dist)
        return [(tidx[i], didx[j]) for i, j in m], [tidx[i] for i in ut], [didx[j] for j in ud]

    def update(self, dets, feats=None, warp=None):
        """dets: (N, 5); feats: (N, D) or None; warp: 2x3 camera-motion matrix of this frame or None.
        Returns [(track_id, box, track)] and keeps `self.new_tracks` (tracks created in this frame)."""
        dets = np.asarray(dets, float).reshape(-1, 5)
        keep = dets[:, 4] >= self.min_conf
        dets = dets[keep]
        if feats is not None:
            feats = np.asarray(feats, float)
            feats = _unit(feats[keep] if feats.ndim == 2 and len(feats) == len(keep)
                          else feats.reshape(len(keep), -1)[keep])
        boxes, confs = dets[:, :4], dets[:, 4]
        for t in self.tracks:
            t.predict(self.kf)
            if warp is not None:
                t.apply_warp(warp)
        all_d = list(range(len(dets)))
        if self.appearance and feats is not None:
            conf_t = [i for i, t in enumerate(self.tracks) if t.state == 'confirmed' and t.feat is not None]
            match = self._cascade_match if self.cascade else self._appearance_match
            m1, um_t, um_d = match(conf_t, boxes, feats, all_d)
            rest = [i for i in range(len(self.tracks)) if i not in set(conf_t)]
            iou_t = rest + [i for i in um_t if self.tracks[i].time_since_update == 1]
            m2, um2, um_d = self._iou_match(iou_t, boxes, um_d)
            matches = m1 + m2
            unmatched_t = [i for i in um_t if i not in set(iou_t)] + um2
        else:
            matches, unmatched_t, um_d = self._iou_match(list(range(len(self.tracks))), boxes, all_d)
        for ti, dj in matches:
            self.tracks[ti].update(self.kf, boxes[dj], confs[dj], None if feats is None else feats[dj])
        for ti in unmatched_t:
            self.tracks[ti].mark_missed()
        for dj in um_d:
            mean, cov = self.kf.initiate(xyxy_to_xyah(boxes[dj]))
            t = Track(self.next_id, mean, cov, confs[dj], None if feats is None else feats[dj],
                      self.n_init, self.max_age, self.ema, self.budget)
            self.next_id += 1
            self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        return [(t.id, t.box, t) for t in self.tracks if t.state == 'confirmed' and t.time_since_update == 0]


# =============================================================================== identity memory
class IdentityMemory:
    """Persistent identity memory (Eqs. 22-25). Non-parametric: one EMA vector per identity, no size limit, no
    expiration. Stored as a growing matrix so that a query is one matrix-vector product, O(K d)."""

    def __init__(self, tau=0.6, beta=0.9):
        self.tau, self.beta = tau, beta
        self.V, self.keys, self.row = None, [], {}

    def __len__(self):
        return len(self.keys)

    def query(self, e, exclude=()):
        """(identity, similarity) of the best stored identity not in `exclude` if cos > tau, else (None, best)."""
        if not self.keys:
            return None, -1.0
        s = self.V[:len(self.keys)] @ e                              # Eq. 23
        if exclude:
            idx = [self.row[k] for k in exclude if k in self.row]
            s[idx] = -np.inf
        i = int(np.argmax(s))
        return (self.keys[i], float(s[i])) if s[i] > self.tau else (None, float(s[i]))   # Eq. 24

    def write(self, k, e):
        e = _unit(np.asarray(e, np.float32))
        if k in self.row:
            i = self.row[k]
            self.V[i] = _unit(self.beta * self.V[i] + (1 - self.beta) * e)            # Eq. 25
            return
        n = len(self.keys)
        if self.V is None:
            self.V = np.zeros((256, len(e)), np.float32)
        elif n == len(self.V):
            self.V = np.concatenate([self.V, np.zeros_like(self.V)])
        self.V[n] = e
        self.keys.append(k)
        self.row[k] = n


class CMAStrongSORT:
    """StrongSORT + identity memory (Algorithm 1). use_memory=False gives StrongSORT with whichever embeddings are fed."""

    def __init__(self, use_memory=True, tau=0.6, beta=0.9, **kw):
        self.base = SortLike(appearance=True, **kw)
        self.use_memory = use_memory
        self.mem = IdentityMemory(tau, beta)
        self.tid2gid, self.next_gid = {}, 1
        self.recovered = 0

    def update(self, dets, feats, warp=None):
        out = self.base.update(dets, feats, warp)
        if not self.use_memory:
            return [(tid, box) for tid, box, _ in out]
        active = {self.tid2gid[t.id] for t in self.base.tracks if t.id in self.tid2gid}
        res = []
        for tid, box, t in out:
            if tid not in self.tid2gid:                      # identity about to be created (Alg. 1, lines 12-17)
                gid, _ = self.mem.query(t.feat, exclude=active) if t.feat is not None else (None, 0)
                if gid is None:
                    gid = self.next_gid
                    self.next_gid += 1
                else:
                    self.recovered += 1
                self.tid2gid[tid] = gid
                active.add(gid)
            gid = self.tid2gid[tid]
            if t.feat is not None:
                self.mem.write(gid, t.feat)                  # Eq. 25 (or a new entry)
            res.append((gid, box))
        return res


class SortTracker:
    def __init__(self, min_conf=0.5, max_age=1, n_init=3, iou_thresh=0.3):
        self.base = SortLike(appearance=False, min_conf=min_conf, max_age=max_age, n_init=n_init,
                             iou_thresh_sort=iou_thresh)

    def update(self, dets, feats=None, warp=None):
        return [(tid, box) for tid, box, _ in self.base.update(dets, None)]


class BoxmotTracker:
    """ByteTrack / OC-SORT from boxmot 25. Detections are passed with class 0; the image is only used for its size."""

    def __init__(self, name, img_shape, **kw):
        import boxmot
        cls = {'ByteTrack': boxmot.ByteTrack, 'OC-SORT': boxmot.OcSort}[name]
        self.t = cls(**kw)
        self.img = np.zeros(img_shape, np.uint8)

    def update(self, dets, feats=None, warp=None):
        d = np.asarray(dets, np.float32).reshape(-1, 5)
        d = np.c_[d, np.zeros(len(d), np.float32)]
        out = np.asarray(self.t.update(d, self.img))
        if out.size == 0:
            return []
        return [(int(r[4]) + 1, r[:4]) for r in out.reshape(len(out), -1)]
