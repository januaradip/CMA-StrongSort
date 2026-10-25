"""
Evaluation with the official TrackEval code (HOTA, CLEAR, Identity) and the tables of the paper.

Every tracker output under WORK/tracks/<ds>_<det>/<tracker>/data/<seq>.txt is evaluated against the validation ground
truth written by the prepare step, with TrackEval's MOTChallenge preprocessing (distractor classes and zero-marked
boxes removed, as on the MOTChallenge server). Reported values are the COMBINED values over the validation sequences
(TrackEval combines the counts of all sequences; they are not averages of per-sequence scores).

    tab:mot17_yolov8, tab:mot20_yolov8, tab:dance_yolov8        YOLOv8   + trackers
    tab:mot17_gswyolo, tab:mot20_gswyolo, tab:dance_gswyolo     GSW-YOLO + trackers
    tab:ablation      ablation A-E (GSW-YOLO, MOT20)
    tab:sensitivity   tau and beta (GSW-YOLO, MOT20)
    tab:efficiency    computational cost (speed step, results/speed_<ds>.csv)
"""
import os

import numpy as np

from . import pipeline as PL

COLS = ['HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW', 'Frag', 'IDs']


def _trackeval():
    for k, v in (('float', float), ('int', int), ('bool', bool)):        # TrackEval still uses the removed aliases
        if not hasattr(np, k):
            setattr(np, k, v)
    import trackeval
    return trackeval


def evaluate_unit(P, ds, det, cores=2):
    """TrackEval on every tracker of one (dataset, detector). Returns a DataFrame (one row per tracker x sequence,
    sequence 'COMBINED' for the whole validation set)."""
    import pandas as pd
    te = _trackeval()
    tdir = P.u('tracks', f'{ds}_{det}')
    trackers = sorted(t for t in os.listdir(tdir) if os.path.exists(os.path.join(tdir, t, 'stats.json')))
    if not trackers:
        return None
    ev_cfg = te.Evaluator.get_default_eval_config()
    ev_cfg.update(USE_PARALLEL=cores > 1, NUM_PARALLEL_CORES=cores, PRINT_RESULTS=False, PRINT_CONFIG=False,
                  TIME_PROGRESS=False, OUTPUT_SUMMARY=True, OUTPUT_DETAILED=False, PLOT_CURVES=False,
                  LOG_ON_ERROR=os.path.join(P.results, 'trackeval_errors.txt'))
    ds_cfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(GT_FOLDER=os.path.join(P.gt, f'{ds}-val'), SEQMAP_FILE=os.path.join(P.gt, 'seqmaps', f'{ds}-val.txt'),
                  TRACKERS_FOLDER=tdir, TRACKERS_TO_EVAL=trackers, SKIP_SPLIT_FOL=True, BENCHMARK=ds,
                  SPLIT_TO_EVAL='val', DO_PREPROC=True, PRINT_CONFIG=False,
                  OUTPUT_FOLDER=os.path.join(P.results, 'trackeval', f'{ds}_{det}'))
    metrics = [te.metrics.HOTA(), te.metrics.CLEAR({'PRINT_CONFIG': False}), te.metrics.Identity({'PRINT_CONFIG': False})]
    # (TrackEval adds the Count metric itself)
    res, _ = te.Evaluator(ev_cfg).evaluate([te.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    rows = []
    for tr, seqs in res['MotChallenge2DBox'].items():
        for seq, r in seqs.items():
            r = r['pedestrian']
            rows.append(dict(dataset=ds, detector=det, tracker=tr, sequence='COMBINED' if seq == 'COMBINED_SEQ' else seq,
                             HOTA=float(np.mean(r['HOTA']['HOTA'])), DetA=float(np.mean(r['HOTA']['DetA'])),
                             AssA=float(np.mean(r['HOTA']['AssA'])), MOTA=float(r['CLEAR']['MOTA']),
                             IDF1=float(r['Identity']['IDF1']), IDSW=int(r['CLEAR']['IDSW']),
                             Frag=int(r['CLEAR']['Frag']), IDs=int(r['Count']['IDs']),
                             GT_IDs=int(r['Count']['GT_IDs'])))
    return pd.DataFrame(rows)


def evaluate_all(P, datasets, detectors, cores=2):
    import pandas as pd
    out = []
    for ds in datasets:
        for det in detectors:
            if not PL.is_done(P.u('tracks', f'{ds}_{det}')):
                PL.log(f'[wait] eval {ds} {det}: tracking not finished')
                continue
            PL.log(f'[run ] TrackEval {ds} {det}')
            df = evaluate_unit(P, ds, det, cores)
            if df is not None:
                df.to_csv(os.path.join(P.results, f'metrics_{ds}_{det}.csv'), index=False)
                out.append(df)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    df.to_csv(os.path.join(P.results, 'metrics_all.csv'), index=False)
    write_tables(P, df)
    return df


# =============================================================================== tables
def _fmt(v, c):
    return f'{v:d}' if c in ('IDSW', 'Frag', 'IDs') else f'{v:.3f}'


def _bold_best(rows, cols):
    """LaTeX cells with the best value of every column in bold."""
    best = {}
    for c in cols:
        vals = [r[c] for r in rows]
        best[c] = min(vals) if c in ('IDSW', 'Frag') else max(vals)
    return [[(f'\\textbf{{{_fmt(r[c], c)}}}' if r[c] == best[c] else _fmt(r[c], c)) for c in cols] for r in rows]


def write_tables(P, df):
    comb = df[df.sequence == 'COMBINED']
    cols = ['HOTA', 'MOTA', 'IDF1', 'IDSW', 'Frag']
    md, tex = ['# Results (validation split, TrackEval)\n'], []
    labels = {'StrongSORT+Con': 'B: + contrastive', 'StrongSORT+Hyb': 'C: + hybrid objective',
              'StrongSORT+Mem': 'D: + memory only', 'StrongSORT': 'A: StrongSORT (baseline)',
              'CMA-StrongSORT': 'E: CMA-StrongSORT'}

    def table(title, sel, names, label=None):
        rows = []
        for n in names:
            r = sel[sel.tracker == n]
            if len(r):
                rows.append(dict(r.iloc[0], name=(label or {}).get(n, n)))
        if not rows:
            return
        md.append(f'\n## {title}\n\n| Method | HOTA | DetA | AssA | MOTA | IDF1 | IDsw | Frag | IDs |\n|' + '---|' * 9)
        for r in rows:
            md.append(f'| {r["name"]} | ' + ' | '.join(_fmt(r[c], c) for c in COLS) + ' |')
        tex.append(f'% {title}')
        for r, cells in zip(rows, _bold_best(rows, cols)):
            name = f'\\textbf{{{r["name"]}}}' if r['name'] in ('CMA-StrongSORT', 'E: CMA-StrongSORT') else r['name']
            tex.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
        tex.append('')

    short = {'MOT17': 'mot17', 'MOT20': 'mot20', 'DanceTrack': 'dance', 'YOLOv8': 'yolov8', 'GSW-YOLO': 'gswyolo'}
    for det in PL.DETECTORS:
        for ds in ('MOT17', 'MOT20', 'DanceTrack'):
            table(f'tab:{short[ds]}_{short[det]} - {ds}, {det} + tracking methods',
                  comb[(comb.dataset == ds) & (comb.detector == det)], PL.MAIN_TRACKERS)
    a = comb[(comb.dataset == PL.CFG['ABL_DS']) & (comb.detector == PL.CFG['ABL_DET'])]
    table(f'tab:ablation - ablation, {PL.CFG["ABL_DS"]}, {PL.CFG["ABL_DET"]}', a,
          ['StrongSORT', 'StrongSORT+Con', 'StrongSORT+Hyb', 'StrongSORT+Mem', 'CMA-StrongSORT'], labels)
    sens_tau = {f'CMA_tau{t:.2f}': f'tau = {t:.2f}' for t in PL.TAUS}
    sens_tau['CMA_tau0.60'] = 'tau = 0.60'
    sens_beta = {f'CMA_beta{b:.2f}': f'beta = {b:.2f}' for b in PL.BETAS}
    a = a.copy()
    base = a[a.tracker == 'CMA-StrongSORT']
    if len(base):                                          # default values come from the CMA-StrongSORT run
        import pandas as pd
        a = pd.concat([a, base.assign(tracker='CMA_tau0.60'), base.assign(tracker='CMA_beta0.90')])
    table('tab:sensitivity (a) - sensitivity to tau (beta = 0.9)', a, list(sens_tau), sens_tau)
    table('tab:sensitivity (b) - sensitivity to beta (tau = 0.6)', a, list(sens_beta), sens_beta)
    open(os.path.join(P.results, 'tables.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(P.results, 'tables.tex'), 'w').write('\n'.join(tex) + '\n')
    PL.log(f'Tables written to {P.results}/tables.md and tables.tex')


def per_sequence_comparison(df, ds, det, a='CMA-StrongSORT', b='StrongSORT'):
    """IDSW, Frag, IDF1 and HOTA of two trackers per validation sequence (the paper reports one run per configuration;
    this shows whether a difference is consistent across sequences)."""
    s = df[(df.dataset == ds) & (df.detector == det) & (df.sequence != 'COMBINED')]
    A = s[s.tracker == a].set_index('sequence')
    B = s[s.tracker == b].set_index('sequence')
    out = A[['HOTA', 'IDF1', 'IDSW', 'Frag']].join(B[['HOTA', 'IDF1', 'IDSW', 'Frag']], lsuffix=f'_{a}', rsuffix=f'_{b}')
    for c in ('HOTA', 'IDF1', 'IDSW', 'Frag'):
        out[f'd{c}'] = out[f'{c}_{a}'] - out[f'{c}_{b}']
    return out
