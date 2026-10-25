"""
Experiment pipeline. Every step works in units that can be run by several Colab sessions at once: a unit is locked
while it runs (LOCK, refreshed while working, considered stale after 3 h), marked with DONE.json when finished and
skipped afterwards. Long units (detector and ReID training) resume where they stopped.

    step        unit                     output (under WORK)
    prepare     dataset                  datasets/ (archives), eval/gt/ (TrackEval ground truth)
    det_train   dataset x detector       detectors/<ds>_<det>/train/weights/best.pt
    detect      dataset x detector       dets/<ds>_<det>/dets.pkl.gz            validation sequences, conf >= 0.1
    reid        dataset x objective      reid/<ds>_<kind>.pt                    ce, hyb (+ con for the ablation)
    embed       dataset x detector       emb/<ds>_<det>/<kind>.pkl.gz, cmc/<ds>.pkl.gz
    track       dataset x detector       tracks/<ds>_<det>/<tracker>/<seq>.txt  MOTChallenge format
    eval        -                        results/*.csv, results/tables.md, results/tables.tex
    speed       dataset                  results/speed_<ds>.csv                 StrongSORT vs CMA-StrongSORT
"""
import gzip, json, os, pickle, shutil, socket, time

import numpy as np

from . import data as D
from . import reid as R
from .tracker import BoxmotTracker, CMAStrongSORT, SortLike, SortTracker

DETECTORS = ['YOLOv8', 'GSW-YOLO']
MAIN_TRACKERS = ['SORT', 'DeepSORT', 'StrongSORT', 'OC-SORT', 'ByteTrack', 'CMA-StrongSORT']
# ablation (Table 9): A = StrongSORT, B = StrongSORT+Con, C = StrongSORT+Hyb, D = StrongSORT+Mem, E = CMA-StrongSORT
ABLATION_TRACKERS = ['StrongSORT+Con', 'StrongSORT+Hyb', 'StrongSORT+Mem']
TAUS = [0.4, 0.5, 0.6, 0.7, 0.8]
BETAS = [0.7, 0.8, 0.85, 0.9, 0.95]
SENS_TRACKERS = [f'CMA_tau{t:.2f}' for t in TAUS if t != 0.6] + [f'CMA_beta{b:.2f}' for b in BETAS if b != 0.9]

CFG = dict(
    # detector (Table 2 / Table 3 of the paper)
    DET_EPOCHS=200, DET_IMGSZ=640, DET_BATCH=16, DET_LR=0.01, DET_MOMENTUM=0.937, DET_PATIENCE=50, DET_WORKERS=2,
    DET_SEED=42, DET_CONF_MIN=0.1, DET_NMS_IOU=0.5, DET_MAX=500, DET_WEIGHTS={},   # e.g. {'MOT17/GSW-YOLO': path}
    # trackers (Table 3)
    HIGH=0.5, MAX_AGE=30, N_INIT=3, MC_LAMBDA=0.98, MAX_COS=0.2, MAX_IOU_DIST=0.7, EMA=0.9, TAU=0.6, BETA=0.9,
    CMC=True, BYTE_MATCH=0.8,
    # ablation / sensitivity setting (Section 5.4 / 5.5)
    ABL_DS='MOT20', ABL_DET='GSW-YOLO',
    # speed (Table 13)
    SPEED_FRAMES_PER_SEQ=300,
    LOCK_STALE_H=3.0,
)
EMB_OF = {'DeepSORT': 'ce', 'StrongSORT': 'ce', 'StrongSORT+Mem': 'ce', 'StrongSORT+Con': 'con',
          'StrongSORT+Hyb': 'hyb', 'CMA-StrongSORT': 'hyb', **{s: 'hyb' for s in SENS_TRACKERS}}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Paths:
    def __init__(self, work, zips=None, local=None):
        self.work = work
        self.zips = zips or os.path.join(work, 'datasets')
        self.local = local or os.path.join(work, 'local')
        self.cache = os.path.join(work, 'cache')
        self.results = os.path.join(work, 'results')
        self.gt = os.path.join(work, 'eval', 'gt')
        for p in (self.cache, self.results):
            os.makedirs(p, exist_ok=True)

    def u(self, *parts):
        return os.path.join(self.work, *parts)

    def root(self, ds):
        return D.fetch(ds, self.zips, self.local)


# =============================================================================== units and locks
def is_done(udir):
    return os.path.exists(os.path.join(udir, 'DONE.json'))


def mark_done(udir, info=None):
    os.makedirs(udir, exist_ok=True)
    info = dict(info or {}, finished=time.strftime('%Y-%m-%d %H:%M:%S'), host=socket.gethostname())
    json.dump(info, open(os.path.join(udir, 'DONE.json'), 'w'), indent=2, default=str)
    lp = os.path.join(udir, 'LOCK')
    if os.path.exists(lp):
        os.remove(lp)


def heartbeat(udir):
    lp = os.path.join(udir, 'LOCK')
    if os.path.exists(lp):
        os.utime(lp)


def try_lock(udir):
    os.makedirs(udir, exist_ok=True)
    lp = os.path.join(udir, 'LOCK')
    if os.path.exists(lp) and time.time() - os.path.getmtime(lp) < CFG['LOCK_STALE_H'] * 3600:
        return False
    json.dump(dict(host=socket.gethostname(), started=time.strftime('%Y-%m-%d %H:%M:%S')), open(lp, 'w'))
    return True


def unlock(udir):
    lp = os.path.join(udir, 'LOCK')
    if os.path.exists(lp):
        os.remove(lp)


def run_unit(udir, name, fn, needs=()):
    """Run fn() for one unit unless it is done, locked by another session or waiting for a prerequisite.
    Returns True when the unit is (now) done."""
    if is_done(udir):
        return True
    missing = [n for n, d in needs if not is_done(d)]
    if missing:
        log(f'[wait] {name}: needs {", ".join(missing)} first')
        return False
    if not try_lock(udir):
        log(f'[skip] {name}: running in another session (LOCK)')
        return False
    log(f'[run ] {name}')
    t0 = time.time()
    try:
        info = fn() or {}
    except BaseException:
        unlock(udir)
        raise
    info['minutes'] = round((time.time() - t0) / 60, 1)
    mark_done(udir, info)
    log(f'[done] {name} ({info["minutes"]} min)')
    return True


def save_gz(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path + '.tmp', 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(path + '.tmp', path)


def load_gz(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def _fp16():
    import torch
    if not torch.cuda.is_available():
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if 'quantize' in DEFAULT_CFG_DICT:
            return {'quantize': 16}
    except ImportError:
        pass
    return {'half': True}


# =============================================================================== prepare
def prepare(P, ds):
    root = P.root(ds)
    D.write_trackeval_gt(ds, root, P.gt)
    return root


# =============================================================================== detectors
def det_dir(P, ds, det):
    return P.u('detectors', f'{ds}_{det}')


def det_weights(P, ds, det):
    w = CFG['DET_WEIGHTS'].get(f'{ds}/{det}')
    if w:
        return w
    wd = os.path.join(det_dir(P, ds, det), 'train', 'weights')
    return os.path.join(wd, 'best.pt') if os.path.exists(os.path.join(wd, 'best.pt')) else os.path.join(wd, 'last.pt')


def train_detector(P, ds, det):
    from ultralytics import YOLO
    from . import gsw_yolo as G
    udir = det_dir(P, ds, det)
    root = P.root(ds)
    yml = D.build_yolo_dataset(ds, root, os.path.join(P.local, 'yolo', ds))
    gsw = det == 'GSW-YOLO'
    trainer = G.make_trainer('yolov8m.pt', os.path.join(udir, 'init_transfer.json')) if gsw else None
    extra = {'trainer': trainer} if gsw else {}
    last = os.path.join(udir, 'train', 'weights', 'last.pt')
    if os.path.exists(last):
        try:
            log('  resuming detector training from last.pt')
            m = YOLO(last)
            m.add_callback('on_fit_epoch_end', lambda tr: heartbeat(udir))
            m.train(resume=True, **extra)
            return _det_info(udir)
        except Exception as e:
            msg = str(e).lower()
            if 'nothing to resume' in msg or 'finished' in msg:
                return _det_info(udir)
            log(f'  last.pt cannot be resumed ({e}); restarting')
            shutil.move(os.path.join(udir, 'train'), os.path.join(udir, f'train_broken_{int(time.time())}'))
    m = YOLO(G.write_variant_yaml('GSW-YOLO', os.path.join(udir, 'yolov8m-GSW-YOLO.yaml'), nc=1)) if gsw \
        else YOLO('yolov8m.pt')
    m.add_callback('on_fit_epoch_end', lambda tr: heartbeat(udir))
    m.train(data=yml, epochs=CFG['DET_EPOCHS'], imgsz=CFG['DET_IMGSZ'], batch=CFG['DET_BATCH'],
            optimizer='SGD', lr0=CFG['DET_LR'], momentum=CFG['DET_MOMENTUM'], patience=CFG['DET_PATIENCE'],
            workers=CFG['DET_WORKERS'], seed=CFG['DET_SEED'], deterministic=True, project=udir, name='train',
            exist_ok=True, plots=True, verbose=False, **extra)
    return _det_info(udir)


def _det_info(udir):
    import csv
    p = os.path.join(udir, 'train', 'results.csv')
    info = {}
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p)))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50-95(B)', 0) or 0))
            info = dict(epochs=len(rows), best_epoch=int(float(best['epoch'])),
                        val_mAP50=float(best['metrics/mAP50(B)']), val_mAP50_95=float(best['metrics/mAP50-95(B)']))
    return info


def detect(P, ds, det):
    from ultralytics import YOLO
    from . import gsw_yolo  # noqa: F401  (registers ShuffleAttention for loading GSW-YOLO checkpoints)
    root = P.root(ds)
    _, va = D.split(ds, root)
    model = YOLO(det_weights(P, ds, det))
    udir = P.u('dets', f'{ds}_{det}')
    out = {}
    for s in va:
        info = D.seqinfo(root, s)
        files = [D.frame_path(root, s, info, f) for f in range(1, info['length'] + 1)]
        out[s] = {}
        for k in range(0, len(files), 32):
            res = model.predict(files[k:k + 32], imgsz=CFG['DET_IMGSZ'], conf=CFG['DET_CONF_MIN'],
                                iou=CFG['DET_NMS_IOU'], max_det=CFG['DET_MAX'], classes=[0], verbose=False, **_fp16())
            for j, r in enumerate(res):
                b = r.boxes
                out[s][k + j + 1] = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()].astype(np.float32)
            heartbeat(udir)
        n = sum(len(v) for v in out[s].values())
        log(f'  {s}: {n} detections ({n / max(1, len(files)):.1f} per frame)')
    save_gz(out, os.path.join(udir, 'dets.pkl.gz'))
    return dict(weights=det_weights(P, ds, det), sequences=va)


# =============================================================================== ReID
def reid_path(P, ds, kind):
    return P.u('reid', f'{ds}_{kind}.pt')


def reid_kinds(ds):
    return ['ce', 'hyb'] + (['con'] if ds == CFG['ABL_DS'] else [])


def train_reid(P, ds, kind):
    root = P.root(ds)
    crops = D.build_reid_crops(ds, root, os.path.join(P.cache, f'{ds}_reid_crops.pkl'))
    udir = P.u('reid', f'{ds}_{kind}')
    R.train(kind, crops['crops'], crops['labels'], R.market_weights(P.cache), reid_path(P, ds, kind),
            log=log, heartbeat=lambda: heartbeat(udir))
    return dict(crops=len(crops['crops']), identities=len(set(crops['labels'])), cfg=dict(R.CFG))


# =============================================================================== embeddings + camera motion
def emb_kinds(ds, det):
    return ['ce', 'hyb'] + (['con'] if (ds, det) == (CFG['ABL_DS'], CFG['ABL_DET']) else [])


def embed(P, ds, det):
    root = P.root(ds)
    dets = load_gz(P.u('dets', f'{ds}_{det}', 'dets.pkl.gz'))
    market = R.market_weights(P.cache)
    encs = {k: R.Encoder(R.load(reid_path(P, ds, k), market)) for k in emb_kinds(ds, det)}
    udir = P.u('emb', f'{ds}_{det}')
    out = {k: {} for k in encs}
    for s in sorted(dets):
        info = D.seqinfo(root, s)
        for k in encs:
            out[k][s] = {}
        for f in range(1, info['length'] + 1):
            d = dets[s].get(f, np.zeros((0, 5), np.float32))
            hi = d[d[:, 4] >= CFG['HIGH']]
            crops = []
            if len(hi):
                img = D.load_rgb(D.frame_path(root, s, info, f))
                crops = [R.crop_resize(img, b) for b in hi[:, :4]]
            for k, e in encs.items():
                out[k][s][f] = e.embed(crops).astype(np.float16)
            if f % 200 == 0:
                heartbeat(udir)
        log(f'  {s}: embeddings {list(encs)}')
    for k in encs:
        save_gz(out[k], os.path.join(udir, f'{k}.pkl.gz'))
    return dict(kinds=list(encs))


def camera_motion(P, ds):
    import cv2
    from boxmot.motion.cmc import get_cmc_method
    root = P.root(ds)
    _, va = D.split(ds, root)
    out = {}
    for s in va:
        info = D.seqinfo(root, s)
        ecc = get_cmc_method('ecc')(warp_mode=cv2.MOTION_EUCLIDEAN, scale=0.15)
        out[s] = {}
        for f in range(1, info['length'] + 1):
            img = cv2.imread(D.frame_path(root, s, info, f))
            out[s][f] = np.asarray(ecc.apply(img, None), np.float32)[:2]
        log(f'  {s}: camera motion of {info["length"]} frames')
    save_gz(out, P.u('cmc', f'{ds}.pkl.gz'))
    return {}


# =============================================================================== tracking
def trackers_for(ds, det):
    t = list(MAIN_TRACKERS)
    if (ds, det) == (CFG['ABL_DS'], CFG['ABL_DET']):
        t += ABLATION_TRACKERS + SENS_TRACKERS
    return t


def make_tracker(name, info):
    c = CFG
    strong = dict(min_conf=c['HIGH'], max_age=c['MAX_AGE'], n_init=c['N_INIT'], mc_lambda=c['MC_LAMBDA'],
                  max_cos_dist=c['MAX_COS'], max_iou_dist=c['MAX_IOU_DIST'], ema=c['EMA'], nsa=True, cascade=False)
    if name == 'SORT':
        return SortTracker(min_conf=c['HIGH'], max_age=1, n_init=c['N_INIT'])
    if name == 'DeepSORT':
        return _Plain(SortLike(appearance=True, min_conf=c['HIGH'], max_age=c['MAX_AGE'], n_init=c['N_INIT'],
                               mc_lambda=1.0, max_cos_dist=c['MAX_COS'], max_iou_dist=c['MAX_IOU_DIST'], ema=None,
                               nsa=False, cascade=True, budget=100))
    if name == 'ByteTrack':
        return BoxmotTracker('ByteTrack', (info['height'], info['width'], 3), min_conf=c['DET_CONF_MIN'],
                             track_thresh=c['HIGH'], match_thresh=c['BYTE_MATCH'], track_buffer=c['MAX_AGE'],
                             frame_rate=int(round(info['fps'])))
    if name == 'OC-SORT':
        return BoxmotTracker('OC-SORT', (info['height'], info['width'], 3), min_conf=c['DET_CONF_MIN'],
                             det_thresh=c['HIGH'], max_age=c['MAX_AGE'], min_hits=c['N_INIT'], iou_threshold=0.3)
    if name in ('StrongSORT', 'StrongSORT+Con', 'StrongSORT+Hyb'):
        return CMAStrongSORT(use_memory=False, **strong)
    if name in ('CMA-StrongSORT', 'StrongSORT+Mem'):
        return CMAStrongSORT(use_memory=True, tau=c['TAU'], beta=c['BETA'], **strong)
    if name.startswith('CMA_tau'):
        return CMAStrongSORT(use_memory=True, tau=float(name[7:]), beta=c['BETA'], **strong)
    if name.startswith('CMA_beta'):
        return CMAStrongSORT(use_memory=True, tau=c['TAU'], beta=float(name[8:]), **strong)
    raise ValueError(name)


class _Plain:
    def __init__(self, base):
        self.base = base

    def update(self, dets, feats=None, warp=None):
        return [(tid, box) for tid, box, _ in self.base.update(dets, feats, warp)]


def uses_cmc(name):
    return CFG['CMC'] and (name.startswith('StrongSORT') or name.startswith('CMA'))


def run_sequence(name, info, dets, emb=None, cmc=None, timer=None):
    """Runs one tracker over one sequence. Returns (rows in MOTChallenge format, stats)."""
    t = make_tracker(name, info)
    rows, K = [], []
    for f in range(1, info['length'] + 1):
        d = dets.get(f, np.zeros((0, 5), np.float32))
        warp = cmc.get(f) if (cmc is not None and uses_cmc(name)) else None
        t0 = time.perf_counter()
        if emb is not None:
            hi = d[d[:, 4] >= CFG['HIGH']]
            out = t.update(hi, np.asarray(emb[f], np.float32), warp)
        else:
            out = t.update(d, None, warp)
        if timer is not None:
            timer.append(time.perf_counter() - t0)
        if hasattr(t, 'mem'):
            K.append(len(t.mem))
        for i, b in out:
            x1, y1, x2, y2 = [float(v) for v in b]
            rows.append(f'{f},{int(i)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1')
    stats = dict(ids=len({r.split(',')[1] for r in rows}), recovered=getattr(t, 'recovered', 0),
                 mean_K=float(np.mean(K)) if K and getattr(t, 'use_memory', False) else None,
                 final_K=K[-1] if K and getattr(t, 'use_memory', False) else None)
    return rows, stats


def track(P, ds, det, trackers=None):
    root = P.root(ds)
    dets = load_gz(P.u('dets', f'{ds}_{det}', 'dets.pkl.gz'))
    cmc = load_gz(P.u('cmc', f'{ds}.pkl.gz')) if CFG['CMC'] else {}
    emb = {k: load_gz(P.u('emb', f'{ds}_{det}', f'{k}.pkl.gz')) for k in emb_kinds(ds, det)}
    udir = P.u('tracks', f'{ds}_{det}')
    allstats = {}
    for name in trackers or trackers_for(ds, det):
        tdir = os.path.join(udir, name)
        sp = os.path.join(tdir, 'stats.json')
        if os.path.exists(sp):
            allstats[name] = json.load(open(sp))
            continue
        st, t0 = {}, time.time()
        for s in sorted(dets):
            info = D.seqinfo(root, s)
            e = emb[EMB_OF[name]][s] if name in EMB_OF else None
            rows, st[s] = run_sequence(name, info, dets[s], e, cmc.get(s))
            os.makedirs(os.path.join(tdir, 'data'), exist_ok=True)
            open(os.path.join(tdir, 'data', f'{s}.txt'), 'w').write('\n'.join(rows) + ('\n' if rows else ''))
            heartbeat(udir)
        st['seconds'] = round(time.time() - t0, 1)
        json.dump(st, open(sp + '.tmp', 'w'), indent=1)
        os.replace(sp + '.tmp', sp)
        allstats[name] = st
        log(f'  {name}: {st["seconds"]} s, identities {sum(v["ids"] for k, v in st.items() if k != "seconds")}')
    return dict(trackers=list(allstats))


# =============================================================================== speed (Table 13)
def speed(P, ds, det='GSW-YOLO'):
    """Per-frame timing of the online pipeline (detect -> embed -> CMC -> associate) for StrongSORT and
    CMA-StrongSORT on the first SPEED_FRAMES_PER_SEQ frames of every validation sequence. Medians, GPU synchronised.
    K-bar is the mean memory size over the full sequences (from the track step)."""
    import cv2, torch
    from ultralytics import YOLO
    from boxmot.motion.cmc import get_cmc_method
    from . import gsw_yolo  # noqa: F401
    root = P.root(ds)
    _, va = D.split(ds, root)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    model = YOLO(det_weights(P, ds, det))
    market = R.market_weights(P.cache)
    rows = []
    for name, kind in (('StrongSORT', 'ce'), ('CMA-StrongSORT', 'hyb')):
        enc = R.Encoder(R.load(reid_path(P, ds, kind), market))
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        T = dict(detect=[], embed=[], cmc=[], assoc=[])
        for s in va:
            info = D.seqinfo(root, s)
            trk = make_tracker(name, info)
            ecc = get_cmc_method('ecc')(warp_mode=cv2.MOTION_EUCLIDEAN, scale=0.15)
            for f in range(1, min(info['length'], CFG['SPEED_FRAMES_PER_SEQ']) + 1):
                bgr = cv2.imread(D.frame_path(root, s, info, f))
                sync(); t0 = time.perf_counter()
                r = model.predict(bgr, imgsz=CFG['DET_IMGSZ'], conf=CFG['DET_CONF_MIN'], iou=CFG['DET_NMS_IOU'],
                                  max_det=CFG['DET_MAX'], classes=[0], verbose=False, **_fp16())[0]
                d = np.c_[r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()]
                sync(); t1 = time.perf_counter()
                hi = d[d[:, 4] >= CFG['HIGH']]
                e = enc.embed([R.crop_resize(bgr[:, :, ::-1], b) for b in hi[:, :4]])
                sync(); t2 = time.perf_counter()
                w = np.asarray(ecc.apply(bgr, None), np.float32)[:2] if CFG['CMC'] else None
                t3 = time.perf_counter()
                trk.update(hi, e, w)
                t4 = time.perf_counter()
                if f > 5:                                            # warm-up frames excluded
                    for k, a, b in (('detect', t0, t1), ('embed', t1, t2), ('cmc', t2, t3), ('assoc', t3, t4)):
                        T[k].append((b - a) * 1000)
            heartbeat(P.u('speed', ds))
        med = {k: float(np.median(v)) for k, v in T.items()}
        st = json.load(open(P.u('tracks', f'{ds}_{det}', name, 'stats.json')))
        Ks = [v['mean_K'] for k, v in st.items() if k != 'seconds' and v.get('mean_K') is not None]
        rows.append(dict(dataset=ds, tracker=name, detect_ms=round(med['detect'], 2), embed_ms=round(med['embed'], 2),
                         cmc_ms=round(med['cmc'], 2), assoc_ms=round(med['assoc'], 3),
                         fps=round(1000 / sum(med.values()), 2),
                         peak_gpu_GB=round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
                         K_bar=round(float(np.mean(Ks)), 1) if Ks else None, frames=len(T['assoc']),
                         gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'))
    # memory query alone (Section 5.7): one query at K = 500 and K = 2000, d = 128
    from .tracker import IdentityMemory
    rng = np.random.default_rng(0)
    q = {}
    for K in (500, 2000):
        m = IdentityMemory()
        for k in range(K):
            m.write(k, rng.standard_normal(128).astype(np.float32))
        e = rng.standard_normal(128).astype(np.float32)
        ts = []
        for _ in range(2000):
            t0 = time.perf_counter(); m.query(e, exclude=()); ts.append(time.perf_counter() - t0)
        q[f'query_ms_K{K}'] = round(float(np.median(ts)) * 1000, 4)
    for r in rows:
        r.update(q)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(P.results, f'speed_{ds}.csv'), index=False)
    return dict(rows=rows)


# =============================================================================== driver
def units(P, ds, det=None):
    return dict(prep=P.u('units', f'prepare_{ds}'), cmc=P.u('cmc', ds),
                reid={k: P.u('reid', f'{ds}_{k}') for k in reid_kinds(ds)},
                det_train=det_dir(P, ds, det) if det else None, detect=P.u('dets', f'{ds}_{det}') if det else None,
                embed=P.u('emb', f'{ds}_{det}') if det else None, track=P.u('tracks', f'{ds}_{det}') if det else None)


def run_step(P, step, datasets, detectors, reverse=False):
    order = lambda xs: list(xs)[::-1] if reverse else list(xs)
    for ds in order(datasets):
        U = units(P, ds)
        if step in ('prepare', 'all'):
            run_unit(U['prep'], f'prepare {ds}', lambda: (prepare(P, ds), {})[1])
            if not is_done(U['prep']):
                continue
        if step in ('reid', 'all'):
            for k in order(reid_kinds(ds)):
                run_unit(U['reid'][k], f'reid {ds} {k}', lambda k=k: train_reid(P, ds, k), [('prepare', U['prep'])])
        if step in ('embed', 'all'):
            run_unit(U['cmc'], f'camera motion {ds}', lambda: camera_motion(P, ds), [('prepare', U['prep'])])
        for det in order(detectors):
            V = units(P, ds, det)
            name = f'{ds} {det}'
            custom = f'{ds}/{det}' in CFG['DET_WEIGHTS']
            if step in ('det_train', 'all') and not custom:
                run_unit(V['det_train'], f'det_train {name}', lambda det=det: train_detector(P, ds, det),
                         [('prepare', U['prep'])])
            dneed = [] if custom else [('det_train ' + name, V['det_train'])]
            if step in ('detect', 'all'):
                run_unit(V['detect'], f'detect {name}', lambda det=det: detect(P, ds, det), dneed)
            if step in ('embed', 'all'):
                need = [('detect ' + name, V['detect'])] + [(f'reid {ds} {k}', U['reid'][k]) for k in emb_kinds(ds, det)]
                run_unit(V['embed'], f'embed {name}', lambda det=det: embed(P, ds, det), need)
            if step in ('track', 'all'):
                need = [('embed ' + name, V['embed'])] + ([('camera motion ' + ds, U['cmc'])] if CFG['CMC'] else [])
                run_unit(V['track'], f'track {name}', lambda det=det: track(P, ds, det), need)
        if step in ('speed',):
            V = units(P, ds, 'GSW-YOLO')
            run_unit(P.u('speed', ds), f'speed {ds}', lambda: speed(P, ds), [('track ' + ds + ' GSW-YOLO', V['track'])])


def status(P, datasets, detectors):
    import pandas as pd
    rows = []

    def st(d):
        if d is None:
            return ''
        if is_done(d):
            return 'done'
        lp = os.path.join(d, 'LOCK')
        if os.path.exists(lp):
            age = (time.time() - os.path.getmtime(lp)) / 60
            return f'running ({age:.0f} min since heartbeat)' if age < CFG['LOCK_STALE_H'] * 60 else 'stale lock'
        return '-'
    for ds in datasets:
        U = units(P, ds)
        rows.append(dict(unit=f'prepare {ds}', state=st(U['prep'])))
        for k, d in U['reid'].items():
            rows.append(dict(unit=f'reid {ds} {k}', state=st(d)))
        rows.append(dict(unit=f'camera motion {ds}', state=st(U['cmc'])))
        for det in detectors:
            V = units(P, ds, det)
            for s in ('det_train', 'detect', 'embed', 'track'):
                rows.append(dict(unit=f'{s} {ds} {det}', state=st(V[s])))
        rows.append(dict(unit=f'speed {ds}', state=st(P.u('speed', ds))))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
