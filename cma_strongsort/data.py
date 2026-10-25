"""
Benchmarks, held-out validation split and data preparation.

Evaluation protocol (Section 5.1.1 of the paper). The MOT17/MOT20 test annotations are not public, so every number is
computed on validation sequences held out from the official TRAINING set. The split is by whole sequences, and the
detector training data and the ReID training crops come only from the training side, so no evaluated sequence or
identity is seen during training. DanceTrack has an official validation set, which is used as is.

    MOT17       train side  MOT17-02, -05, -10, -11        validation  MOT17-04, -09, -13
    MOT20       train side  MOT20-02, -05                  validation  MOT20-01, -03
    DanceTrack  train side  official train (40 seq.)       validation  official val (25 seq.)

MOT17 is distributed with three copies of every sequence (DPM, FRCNN, SDP public detections); the images and the
ground truth are identical, so only the FRCNN copy is used and the sequence is called MOT17-XX.

Ground truth: MOT17/MOT20 boxes with class 1 (pedestrian) and flag 1 are the targets. For evaluation the full gt.txt
is given to TrackEval, which removes distractor classes and zero-marked boxes itself. DanceTrack has one class.
"""
import glob, io, json, os, re, shutil, subprocess, zipfile

import numpy as np

SPLITS = {
    'MOT17': dict(train=['MOT17-02', 'MOT17-05', 'MOT17-10', 'MOT17-11'], val=['MOT17-04', 'MOT17-09', 'MOT17-13']),
    'MOT20': dict(train=['MOT20-02', 'MOT20-05'], val=['MOT20-01', 'MOT20-03']),
    'DanceTrack': dict(train='official', val='official'),
}
DATASETS = list(SPLITS)
URLS = {'MOT17': 'https://motchallenge.net/data/MOT17.zip', 'MOT20': 'https://motchallenge.net/data/MOT20.zip'}
DANCETRACK_HF = 'noahcao/dancetrack'

# detector training data: every n-th frame of the training sequences; the last 10 % of every training sequence is
# the detector's own validation set (model selection), so the held-out sequences are never used for it
DET_STRIDE = {'MOT17': 1, 'MOT20': 2, 'DanceTrack': 10}
# ReID crops: every n-th frame, visibility >= 0.5, height >= 32 px, at most N crops per identity (evenly spaced)
REID_STRIDE = {'MOT17': 3, 'MOT20': 5, 'DanceTrack': 10}
REID_MAX_PER_ID = 30


def log(*a):
    print(*a, flush=True)


# =============================================================================== download / extraction
def _download(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    log(f'Downloading {url} ...')
    subprocess.run(['wget', '-q', '--show-progress', '-c', '-O', tmp, url], check=True)
    os.replace(tmp, dst)


def _zip_ok(p):
    try:
        with zipfile.ZipFile(p) as z:
            return len(z.namelist()) > 0
    except Exception:
        return False


def fetch(ds, zip_dir, local):
    """Makes `local/<ds>/` ready. Archives are looked for in (and downloaded once to) zip_dir, e.g. a Drive folder."""
    root = os.path.join(local, ds)
    if os.path.exists(os.path.join(root, 'READY')):
        return root
    os.makedirs(root, exist_ok=True)
    if ds in URLS:
        z = os.path.join(zip_dir, f'{ds}.zip')
        if not _zip_ok(z):
            _download(URLS[ds], z)
        tmp = os.path.join(local, f'_{ds}')
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if ('/train/' in n or n.startswith('train/'))
                     and (ds != 'MOT17' or '-FRCNN/' in n)]
            log(f'Extracting {len(names)} files of {ds}/train ...')
            f.extractall(tmp, names)
        os.makedirs(os.path.join(root, 'sequences'), exist_ok=True)
        for ini in sorted(glob.glob(os.path.join(tmp, '**', 'seqinfo.ini'), recursive=True)):
            d = os.path.dirname(ini)
            shutil.move(d, os.path.join(root, 'sequences', os.path.basename(d).replace('-FRCNN', '')))
        shutil.rmtree(tmp, ignore_errors=True)
    else:                                                       # DanceTrack: train*.zip and val*.zip
        zips = sorted(glob.glob(os.path.join(zip_dir, 'DanceTrack', '*.zip')))
        if not [z for z in zips if os.path.basename(z).startswith('val')]:
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                files = [f for f in list_repo_files(DANCETRACK_HF, repo_type='dataset')
                         if re.match(r'^(train\d*|val)\.zip$', os.path.basename(f))]
                if not files:
                    raise FileNotFoundError('no train*.zip / val.zip in the repository')
                log(f'Downloading DanceTrack {files} from huggingface.co/datasets/{DANCETRACK_HF} ...')
                for f in files:
                    p = hf_hub_download(DANCETRACK_HF, f, repo_type='dataset',
                                        local_dir=os.path.join(zip_dir, 'DanceTrack'))
                    log('  ', p)
            except Exception as e:
                raise RuntimeError(
                    f'DanceTrack could not be downloaded automatically ({e}). Download train1.zip, train2.zip and '
                    f'val.zip from https://dancetrack.github.io and put them in {os.path.join(zip_dir, "DanceTrack")}')
            zips = sorted(glob.glob(os.path.join(zip_dir, 'DanceTrack', '**', '*.zip'), recursive=True))
        tmp = os.path.join(local, '_dt')
        for z in zips:
            part = 'val' if os.path.basename(z).startswith('val') else 'train'
            log(f'Extracting {os.path.basename(z)} ...')
            with zipfile.ZipFile(z) as f:
                f.extractall(os.path.join(tmp, part))
        os.makedirs(os.path.join(root, 'sequences'), exist_ok=True)
        side = {}
        for part in ('train', 'val'):
            for ini in glob.glob(os.path.join(tmp, part, '**', 'seqinfo.ini'), recursive=True):
                d = os.path.dirname(ini)
                name = os.path.basename(d)
                shutil.move(d, os.path.join(root, 'sequences', name))
                side[name] = part
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump(side, open(os.path.join(root, 'dancetrack_sides.json'), 'w'), indent=1)
    check(ds, root)
    open(os.path.join(root, 'READY'), 'w').write('ok')
    return root


def check(ds, root):
    tr, va = split(ds, root)
    missing = [s for s in tr + va if not os.path.exists(os.path.join(root, 'sequences', s, 'gt', 'gt.txt'))]
    if missing:
        raise FileNotFoundError(f'{ds}: sequences without gt/gt.txt: {missing}')
    log(f'{ds}: {len(tr)} training-side and {len(va)} validation sequences')


# =============================================================================== sequences
def split(ds, root):
    s = SPLITS[ds]
    if s['train'] != 'official':
        return list(s['train']), list(s['val'])
    side = json.load(open(os.path.join(root, 'dancetrack_sides.json')))
    return sorted(k for k, v in side.items() if v == 'train'), sorted(k for k, v in side.items() if v == 'val')


def seqinfo(root, seq):
    info = {}
    for line in open(os.path.join(root, 'sequences', seq, 'seqinfo.ini')):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            info[k.strip()] = v.strip()
    return dict(name=seq, fps=float(info.get('frameRate', 30)), length=int(info['seqLength']),
                width=int(info['imWidth']), height=int(info['imHeight']), ext=info.get('imExt', '.jpg'),
                imdir=info.get('imDir', 'img1'))


def frame_path(root, seq, info, f):
    return os.path.join(root, 'sequences', seq, info['imdir'], f'{f:06d}{info["ext"]}')


def read_gt(root, seq, ds):
    """Targets as {frame: (N, 6) [id, x1, y1, x2, y2, visibility]}: class 1 (pedestrian), flag 1."""
    a = np.loadtxt(os.path.join(root, 'sequences', seq, 'gt', 'gt.txt'), delimiter=',', ndmin=2)
    if a.shape[1] >= 8:
        a = a[(a[:, 6] == 1) & (a[:, 7] == 1)]
    vis = a[:, 8] if a.shape[1] >= 9 else np.ones(len(a))
    out = {}
    for r, v in zip(a, vis):
        out.setdefault(int(r[0]), []).append([r[1], r[2], r[3], r[2] + r[4], r[3] + r[5], v])
    return {k: np.array(v, np.float32) for k, v in out.items()}


def load_rgb(path):
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1]


# =============================================================================== TrackEval ground truth
def write_trackeval_gt(ds, root, gt_root):
    """GT folder in the MOTChallenge layout expected by TrackEval: <gt_root>/<ds>-val/<seq>/{gt/gt.txt, seqinfo.ini}
    and <gt_root>/seqmaps/<ds>-val.txt."""
    _, va = split(ds, root)
    for s in va:
        d = os.path.join(gt_root, f'{ds}-val', s)
        os.makedirs(os.path.join(d, 'gt'), exist_ok=True)
        shutil.copy(os.path.join(root, 'sequences', s, 'gt', 'gt.txt'), os.path.join(d, 'gt', 'gt.txt'))
        shutil.copy(os.path.join(root, 'sequences', s, 'seqinfo.ini'), os.path.join(d, 'seqinfo.ini'))
    os.makedirs(os.path.join(gt_root, 'seqmaps'), exist_ok=True)
    with open(os.path.join(gt_root, 'seqmaps', f'{ds}-val.txt'), 'w') as f:
        f.write('name\n' + ''.join(f'{s}\n' for s in va))


# =============================================================================== detector training data
def build_yolo_dataset(ds, root, out):
    """YOLO-format person dataset from the TRAINING-side sequences: images are symlinks, labels class 0.
    Returns the data yaml path."""
    yml = os.path.join(out, 'data.yaml')
    if os.path.exists(yml):
        return yml
    tr, _ = split(ds, root)
    lists = {'train': [], 'val': []}
    for s in tr:
        info = seqinfo(root, s)
        gt = read_gt(root, s, ds)
        frames = list(range(1, info['length'] + 1))[::DET_STRIDE[ds]]
        n_val = max(1, int(round(0.1 * len(frames))))
        for i, f in enumerate(frames):
            part = 'val' if i >= len(frames) - n_val else 'train'
            name = f'{s}_{f:06d}'
            img = os.path.join(out, 'images', part, name + info['ext'])
            lab = os.path.join(out, 'labels', part, name + '.txt')
            os.makedirs(os.path.dirname(img), exist_ok=True); os.makedirs(os.path.dirname(lab), exist_ok=True)
            if not os.path.lexists(img):
                os.symlink(frame_path(root, s, info, f), img)
            W, H = info['width'], info['height']
            rows = []
            for _, x1, y1, x2, y2, v in gt.get(f, []):
                if v <= 0:
                    continue
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                rows.append(f'0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}')
            open(lab, 'w').write('\n'.join(rows) + ('\n' if rows else ''))
            lists[part].append(img)
    with open(yml, 'w') as f:
        f.write(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n')
    log(f'{ds}: detector data {len(lists["train"])} train / {len(lists["val"])} val images '
        f'(stride {DET_STRIDE[ds]}, training-side sequences only)')
    return yml


# =============================================================================== ReID crops
def build_reid_crops(ds, root, out_path):
    """JPEG crops (256x128) of the training-side ground truth. Identity label = '<seq>:<track id>'."""
    import pickle
    from PIL import Image
    from . import reid as R
    if os.path.exists(out_path):
        return pickle.load(open(out_path, 'rb'))
    tr, _ = split(ds, root)
    crops, labels = [], []
    for s in tr:
        info = seqinfo(root, s)
        gt = read_gt(root, s, ds)
        want = {}
        for f in range(1, info['length'] + 1, REID_STRIDE[ds]):
            for tid, x1, y1, x2, y2, v in gt.get(f, []):
                if v >= 0.5 and y2 - y1 >= 32:
                    want.setdefault(int(tid), []).append((f, (x1, y1, x2, y2)))
        by_frame = {}
        for tid, lst in want.items():
            if len(lst) < 2:
                continue
            idx = np.unique(np.linspace(0, len(lst) - 1, min(REID_MAX_PER_ID, len(lst))).round().astype(int))
            for i in idx:
                f, b = lst[i]
                by_frame.setdefault(f, []).append((tid, b))
        for f in sorted(by_frame):
            img = load_rgb(frame_path(root, s, info, f))
            for tid, b in by_frame[f]:
                buf = io.BytesIO()
                Image.fromarray(R.crop_resize(img, b)).save(buf, 'JPEG', quality=92)
                crops.append(buf.getvalue()); labels.append(f'{s}:{tid}')
        log(f'  {s}: {sum(1 for l in labels if l.startswith(s + ":"))} crops')
    data = dict(crops=crops, labels=labels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pickle.dump(data, open(out_path + '.tmp', 'wb'))
    os.replace(out_path + '.tmp', out_path)
    log(f'{ds}: {len(crops)} ReID crops of {len(set(labels))} identities')
    return data
