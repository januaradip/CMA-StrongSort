# CMA-StrongSORT: Contrastive Memory-Augmented StrongSORT for Robust Long-Term Person Tracking

Januar Adi Putra, Nanik Suciati, Chastine Fatichah — Department of Informatics, Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia

Code for the paper. It contains everything needed to reproduce the experiments: detector training (YOLOv8 and GSW-YOLO), training of the re-identification models, all trackers, evaluation with TrackEval, and the scripts that build the tables.

## Method

CMA-StrongSORT extends StrongSORT in two ways.

1. **Contrastive re-identification.** The backbone is OSNet-x1.0, initialised from Market-1501. A 128-d head with BatchNorm and L2 normalisation (Eq. 16) is added on top. It is fine-tuned on crops from the training sequences with `L = L_con + λ L_tri`: margin m = 1.0, α = 0.3, λ = 1.0 (Eqs. 17–19).
2. **Persistent identity memory.** The memory stores one EMA vector per identity (β = 0.9, Eq. 25). It has no size limit and no expiration. When a new identity is about to be created, its embedding is first compared with the stored identities that are not currently active (Eq. 23). If cos > τ = 0.6 (Eq. 24), the old identity is reused (Algorithm 1). The memory is non-parametric: it has no trainable weights.

Association follows StrongSORT:
- NSA Kalman filter, EMA track features, and ECC camera-motion compensation;
- cost `C = λ C_app + (1 − λ) C_motion` with λ = 0.98 (γ = 0.02 in Eq. 21), Mahalanobis gating, maximum cosine distance 0.2;
- then IoU matching (maximum IoU distance 0.7);
- max age 30, n_init 3, detection threshold 0.5.

| File | Content |
|---|---|
| `cma_strongsort/tracker.py` | SORT, DeepSORT, StrongSORT, CMA-StrongSORT, identity memory; ByteTrack / OC-SORT wrappers (BoxMOT) |
| `cma_strongsort/reid.py` | OSNet ReID models and their training (classification, contrastive, contrastive + triplet) |
| `cma_strongsort/gsw_yolo.py` | GSW-YOLO (Ghost backbone, Shuffle Attention neck, Wise-IoU v3) on Ultralytics |
| `cma_strongsort/data.py` | Datasets, held-out validation split, detector and ReID training data |
| `cma_strongsort/pipeline.py` | Experiment steps (resumable, several Colab sessions in parallel) |
| `cma_strongsort/evaluate.py` | TrackEval evaluation and the tables of the paper |
| `run.py` | Command-line entry point |
| `notebooks/CMA_StrongSORT_Colab.ipynb` | One-click Colab notebook |

## Evaluation protocol

The MOT17/MOT20 test annotations are not public, so all results are computed on whole validation sequences held out from the official training sets. The detectors and the ReID models are trained only on the training side, so no evaluated sequence or identity is seen during training.

| Benchmark | Training side | Validation |
|---|---|---|
| MOT17 (FRCNN copy) | 02, 05, 10, 11 | 04, 09, 13 |
| MOT20 | 02, 05 | 01, 03 |
| DanceTrack | official train (40 sequences) | official val (25 sequences) |

- **Metrics.** HOTA, DetA, AssA, MOTA, IDF1, IDsw, and Frag come from the official [TrackEval](https://github.com/JonathonLuiten/TrackEval) with MOTChallenge preprocessing. Values are combined over the validation sequences.
- **Same inputs for every tracker.** All trackers receive the same detections (conf ≥ 0.1; ByteTrack uses the low-score band, the others use conf ≥ 0.5) and run on the same sequences.
- **Same ReID data and schedule.** DeepSORT and StrongSORT use OSNet fine-tuned with a classification objective on the same crops and with the same schedule as the contrastive models, so the ablation compares training objectives under identical data.

| Paper table | Content | Output |
|---|---|---|
| `tab:mot17_yolov8`, `tab:mot20_yolov8`, `tab:dance_yolov8` | YOLOv8 + SORT, DeepSORT, StrongSORT, OC-SORT, ByteTrack, CMA-StrongSORT | `results/tables.md`, `results/tables.tex` |
| `tab:mot17_gswyolo`, `tab:mot20_gswyolo`, `tab:dance_gswyolo` | GSW-YOLO + the same trackers | idem |
| `tab:ablation` | A StrongSORT, B + contrastive, C + contrastive & triplet, D + memory only, E CMA-StrongSORT (MOT20, GSW-YOLO) | idem |
| `tab:sensitivity` | τ ∈ {0.4 … 0.8}, β ∈ {0.7 … 0.95} (MOT20, GSW-YOLO) | idem |
| `tab:efficiency` | detection / embedding / CMC / association time, FPS, peak GPU memory, mean memory size K̄ | `results/speed_<dataset>.csv` |

Per-sequence values are in `results/metrics_all.csv`. TrackEval's own summaries are in `results/trackeval/`.

## Reproducing

### Google Colab (recommended)

1. Open `notebooks/CMA_StrongSORT_Colab.ipynb` in Colab and set `WORK` to a Google Drive folder.
2. Run all cells with `LANGKAH = 'all'`. The notebook installs the pinned packages and downloads MOT17 and MOT20 (motchallenge.net) and DanceTrack (Hugging Face) once to `WORK/datasets`. Then it runs every unit that is not finished yet.
3. Several sessions can run at once. Set `REVERSE = True` in the second session so that it starts from the other end.
4. Interrupted units resume in the next session. Detector training and ReID training continue from their last epoch.
5. When everything is done, run `LANGKAH = 'eval'` (CPU is enough) and `LANGKAH = 'speed'` (on the GPU used for the paper, a T4).

### Command line

```bash
pip install -r requirements.txt
python run.py --work /path/to/work --step all            # every remaining unit
python run.py --work /path/to/work --step status         # progress
python run.py --work /path/to/work --step eval           # TrackEval + tables
python run.py --work /path/to/work --step speed          # timing table
```

Use `--datasets MOT17 MOT20` and `--detectors GSW-YOLO` to limit the run. Use `--set KEY=VALUE` to change a setting in `pipeline.CFG`, e.g. `--set DET_EPOCHS=100`. Detector weights trained elsewhere can be used through `CFG['DET_WEIGHTS']`, e.g. `{'MOT17/GSW-YOLO': 'best.pt'}`.

### Approximate compute (T4)

| Step | Units | Time per unit |
|---|---|---|
| Detector training (200 epochs, 640 px) | 6 (3 datasets × 2 detectors) | 3–6 h (A100: about 3× faster) |
| ReID training (60 epochs) | 7 | 15–60 min |
| Detection + embeddings + tracking | 6 | 10–40 min |
| Evaluation | 1 | 10–30 min (CPU) |

### Environment of the paper

Google Colab, NVIDIA T4, `ultralytics==8.4.183`, `boxmot==25.0.0`, TrackEval (master), seed 42; both detectors at the m scale (YOLOv8m, and GSW-YOLO at YOLOv8m width and depth, initialised from the COCO-pretrained yolov8m.pt). The notebook prints the Python, PyTorch, CUDA and GPU versions of each session (`results/environment.txt`).

## Citation

```bibtex
@article{putra_cma_strongsort,
  title   = {CMA-StrongSORT: A Contrastive Memory-Augmented StrongSORT for Robust Long-Term Person Tracking},
  author  = {Putra, Januar Adi and Suciati, Nanik and Fatichah, Chastine},
  note    = {under review}
}
```

## Acknowledgements

This work was supported by the Indonesian Endowment Fund for Education (LPDP). It builds on [Ultralytics](https://github.com/ultralytics/ultralytics), [BoxMOT](https://github.com/mikel-brostrom/boxmot), [TrackEval](https://github.com/JonathonLuiten/TrackEval), and [OSNet](https://github.com/KaiyangZhou/deep-person-reid). Ultralytics and BoxMOT are released under AGPL-3.0, so this repository should be distributed under a compatible license.
