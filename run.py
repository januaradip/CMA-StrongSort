#!/usr/bin/env python
"""
CMA-StrongSORT experiments (MOT17, MOT20, DanceTrack; YOLOv8 and GSW-YOLO detectors).

    python run.py --work WORK --step all                      # everything that is not done yet
    python run.py --work WORK --step status
    python run.py --work WORK --step eval                     # TrackEval + tables (CPU is enough)
    python run.py --work WORK --step speed                    # timing table (GPU)

Steps: prepare, det_train, detect, reid, embed, track, all, eval, speed, status.
Several sessions can run the same command at once (use --reverse in the second one); units are locked.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    from cma_strongsort import data as D, pipeline as PL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True, help='output folder (e.g. on Google Drive)')
    ap.add_argument('--zips', help='folder with / for the dataset archives (default WORK/datasets)')
    ap.add_argument('--local', help='fast local folder for the extracted datasets (default WORK/local)')
    ap.add_argument('--step', required=True,
                    choices=['prepare', 'det_train', 'detect', 'reid', 'embed', 'track', 'all', 'eval', 'speed', 'status'])
    ap.add_argument('--datasets', nargs='*', default=D.DATASETS)
    ap.add_argument('--detectors', nargs='*', default=PL.DETECTORS)
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override pipeline.CFG, e.g. DET_EPOCHS=100')
    a = ap.parse_args(argv)
    for kv in a.set:
        k, v = kv.split('=', 1)
        if k not in PL.CFG:
            raise SystemExit(f'unknown setting {k}')
        PL.CFG[k] = type(PL.CFG[k])(v) if not isinstance(PL.CFG[k], (bool, dict)) else eval(v)
    P = PL.Paths(a.work, a.zips, a.local)
    if a.step == 'status':
        PL.status(P, a.datasets, a.detectors)
    elif a.step == 'eval':
        from cma_strongsort import evaluate as E
        df = E.evaluate_all(P, a.datasets, a.detectors)
        if df is not None:
            print(open(os.path.join(P.results, 'tables.md')).read())
    else:
        PL.run_step(P, a.step, a.datasets, a.detectors, a.reverse)
        PL.status(P, a.datasets, a.detectors)


if __name__ == '__main__':
    main()
