"""
GSW-YOLO: YOLOv8m with Ghost modules in the backbone, Shuffle Attention in the neck and Wise-IoU v3 as
bounding-box regression loss, built on Ultralytics.

    Ghost   : GhostConv (stride-2 downsampling) and C3Ghost blocks replace Conv and C2f in the backbone
              (Han et al., GhostNet, CVPR 2020).
    SA      : Shuffle Attention on the three neck outputs (P3, P4, P5) before the Detect head
              (Zhang and Yang, SA-Net, ICASSP 2021).
    WIoU v3 : dynamic non-monotonic focusing IoU loss replacing CIoU (Tong et al., Wise-IoU, 2023).

The ablation variants share the same code:

    'YOLOv8m'        reference (standard Ultralytics model, trained with the baseline code)
    'v8m-Ghost'      + Ghost backbone                      (CIoU)
    'v8m-Ghost-SA'   + Ghost backbone + Shuffle Attention  (CIoU)
    'GSW-YOLO'       + Ghost backbone + Shuffle Attention + WIoU v3

Initialisation: every tensor whose name and shape match is copied from the COCO-pretrained yolov8m.pt
(the same file the YOLOv8m baseline starts from). Layers that do not exist in YOLOv8m (Ghost blocks, SA)
start from random initialisation. A COCO-pretrained GSW-YOLO checkpoint can be given instead.
"""
import copy, json, math, os

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn import tasks
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import BboxLoss

try:
    from ultralytics.utils import RANK
except ImportError:          # pragma: no cover
    RANK = -1


# =============================================================================== Shuffle Attention
class ShuffleAttention(nn.Module):
    """Shuffle Attention (SA-Net). Channels are split into G groups; each group is split in half, one half gets
    channel attention (global average pooling), the other spatial attention (group normalisation); the halves
    are concatenated and the channels shuffled so that information flows between groups.
    Input and output have the same shape."""

    def __init__(self, channels, groups=8):
        super().__init__()
        if channels % (2 * groups):
            raise ValueError(f'ShuffleAttention: channels ({channels}) must be divisible by 2*groups ({2 * groups})')
        self.channels, self.groups = channels, groups
        c = channels // (2 * groups)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(c, c)
        self.cweight = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, c, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, c, 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        return x.reshape(b, groups, c // groups, h, w).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise RuntimeError(f'ShuffleAttention built for {self.channels} channels, got {c}')
        x = x.reshape(b * self.groups, -1, h, w)
        x0, x1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x0)                                             # channel attention
        xn = x0 * self.sigmoid(self.cweight * xn + self.cbias)
        xs = self.gn(x1)                                                   # spatial attention
        xs = x1 * self.sigmoid(self.sweight * xs + self.sbias)
        out = torch.cat([xn, xs], dim=1).reshape(b, -1, h, w)
        return self.channel_shuffle(out, 2)


def register():
    """Make the custom layers known to the Ultralytics YAML parser."""
    tasks.ShuffleAttention = ShuffleAttention


register()


# =============================================================================== Wise-IoU v3
class WIoUv3BboxLoss(BboxLoss):
    """Ultralytics BboxLoss with the CIoU term replaced by Wise-IoU v3:

        L_IoU   = 1 - IoU
        R_WIoU  = exp( ((x - x_gt)^2 + (y - y_gt)^2) / (W_g^2 + H_g^2)* )      (* = detached, W_g/H_g enclosing box)
        beta    = L_IoU* / mean(L_IoU)                                          (running mean, momentum m)
        r       = beta / (delta * alpha^(beta - delta)),  alpha = 1.9, delta = 3
        L_WIoU  = r * R_WIoU * L_IoU

    The DFL term is unchanged. The running mean is only updated during training (gradient enabled)."""

    def __init__(self, reg_max=16, alpha=1.9, delta=3.0, momentum=1 - 0.5 ** (1 / 7000)):
        super().__init__(reg_max)
        self.alpha, self.delta, self.momentum = alpha, delta, momentum
        self.register_buffer('iou_mean', torch.tensor(1.0))

    def wiou(self, pred, target):
        """pred, target: (N, 4) xyxy. Returns the per-box WIoU v3 loss (N, 1)."""
        eps = 1e-7
        px1, py1, px2, py2 = pred.unbind(-1)
        tx1, ty1, tx2, ty2 = target.unbind(-1)
        inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
        union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter + eps
        l_iou = 1 - inter / union
        cw = torch.max(px2, tx2) - torch.min(px1, tx1)                    # enclosing box
        ch = torch.max(py2, ty2) - torch.min(py1, ty1)
        dist = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4
        r_wiou = torch.exp(dist / (cw ** 2 + ch ** 2 + eps).detach())
        if torch.is_grad_enabled() and l_iou.numel():
            with torch.no_grad():
                self.iou_mean.mul_(1 - self.momentum).add_(self.momentum * l_iou.detach().mean())
        beta = l_iou.detach() / self.iou_mean.clamp(min=eps)
        r = beta / (self.delta * self.alpha ** (beta - self.delta))
        return (r * r_wiou * l_iou).unsqueeze(-1)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, *rest, **kw):
        _, loss_dfl = super().forward(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                      target_scores_sum, fg_mask, *rest, **kw)
        idx = fg_mask.nonzero(as_tuple=True) if fg_mask.dtype == torch.bool else fg_mask
        weight = target_scores[idx].sum(-1, keepdim=True)
        loss_iou = (self.wiou(pred_bboxes[idx].float(), target_bboxes[idx].float()) * weight).sum() / target_scores_sum
        return loss_iou, loss_dfl


class GSWDetectionModel(DetectionModel):
    """DetectionModel whose bounding-box loss follows the 'iou_loss' key of the model YAML ('ciou' or 'wiou')."""

    def init_criterion(self):
        crit = super().init_criterion()
        if self.yaml.get('iou_loss', 'ciou') == 'wiou':
            if not hasattr(crit, 'bbox_loss'):
                raise RuntimeError('This Ultralytics version has no bbox_loss in the detection loss; WIoU cannot be set')
            dev = next(self.parameters()).device
            crit.bbox_loss = WIoUv3BboxLoss(self.model[-1].reg_max).to(dev)
            if not getattr(GSWDetectionModel, '_announced', False):
                print('Bounding-box loss: Wise-IoU v3 (alpha 1.9, delta 3)', flush=True)
                GSWDetectionModel._announced = True
        return crit


# =============================================================================== architectures
_V8_BACKBONE = [
    [-1, 1, 'Conv', [64, 3, 2]],            # 0-P1/2
    [-1, 1, 'Conv', [128, 3, 2]],           # 1-P2/4
    [-1, 3, 'C2f', [128, True]],
    [-1, 1, 'Conv', [256, 3, 2]],           # 3-P3/8
    [-1, 6, 'C2f', [256, True]],
    [-1, 1, 'Conv', [512, 3, 2]],           # 5-P4/16
    [-1, 6, 'C2f', [512, True]],
    [-1, 1, 'Conv', [1024, 3, 2]],          # 7-P5/32
    [-1, 3, 'C2f', [1024, True]],
    [-1, 1, 'SPPF', [1024, 5]],             # 9
]
_GHOST_BACKBONE = [
    [-1, 1, 'Conv', [64, 3, 2]],            # 0-P1/2   (stem kept as a standard convolution)
    [-1, 1, 'GhostConv', [128, 3, 2]],      # 1-P2/4
    [-1, 3, 'C3Ghost', [128, True]],
    [-1, 1, 'GhostConv', [256, 3, 2]],      # 3-P3/8
    [-1, 6, 'C3Ghost', [256, True]],
    [-1, 1, 'GhostConv', [512, 3, 2]],      # 5-P4/16
    [-1, 6, 'C3Ghost', [512, True]],
    [-1, 1, 'GhostConv', [1024, 3, 2]],     # 7-P5/32
    [-1, 3, 'C3Ghost', [1024, True]],
    [-1, 1, 'SPPF', [1024, 5]],             # 9
]
_V8_NECK = [
    [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
    [[-1, 6], 1, 'Concat', [1]],            # cat backbone P4
    [-1, 3, 'C2f', [512]],                  # 12
    [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
    [[-1, 4], 1, 'Concat', [1]],            # cat backbone P3
    [-1, 3, 'C2f', [256]],                  # 15 (P3/8)
    [-1, 1, 'Conv', [256, 3, 2]],
    [[-1, 12], 1, 'Concat', [1]],           # cat head P4
    [-1, 3, 'C2f', [512]],                  # 18 (P4/16)
    [-1, 1, 'Conv', [512, 3, 2]],
    [[-1, 9], 1, 'Concat', [1]],            # cat head P5
    [-1, 3, 'C2f', [1024]],                 # 21 (P5/32)
]
SCALE_M = (0.67, 0.75, 768)                 # depth, width, max channels of YOLOv8m
SA_GROUPS = 8

VARIANTS = {
    #  name            ghost  sa     iou loss
    'v8m-Ghost':     (True,  False, 'ciou'),
    'v8m-Ghost-SA':  (True,  True,  'ciou'),
    'GSW-YOLO':      (True,  True,  'wiou'),
    'v8m-SA':        (False, True,  'ciou'),   # not in the run plan; available for extra ablations
    'v8m-WIoU':      (False, False, 'wiou'),   # idem
}


def _width(c):
    d, w, mc = SCALE_M
    return int(math.ceil(min(c, mc) * w / 8) * 8)


def variant_cfg(name, nc=6):
    """Model YAML (as dict) of an ablation variant at YOLOv8m scale."""
    ghost, sa, iou = VARIANTS[name]
    head = copy.deepcopy(_V8_NECK)
    outs = [15, 18, 21]
    if sa:
        for i, c in zip(outs, (256, 512, 1024)):
            head.append([i, 1, 'ShuffleAttention', [_width(c), SA_GROUPS]])
        outs = [22, 23, 24]
    head.append([outs, 1, 'Detect', ['nc']])
    return dict(nc=nc, scales={'m': list(SCALE_M)}, scale='m',
                backbone=copy.deepcopy(_GHOST_BACKBONE if ghost else _V8_BACKBONE), head=head,
                iou_loss=iou, variant=name)


def write_variant_yaml(name, path, nc=6):
    import yaml
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(variant_cfg(name, nc), f, sort_keys=False)
    return path


# =============================================================================== initialisation
def transfer_weights(model, source='yolov8m.pt'):
    """Copy every tensor of `source` whose name and shape match into `model`. The Detect layer is matched by
    position (it is the last layer in both models, but its index differs when SA layers are added).
    Returns a summary that is stored with the run."""
    from ultralytics import YOLO
    src = YOLO(source).model.float()
    sd_src = src.state_dict()
    n_src, n_dst = len(src.model) - 1, len(model.model) - 1
    remapped = {}
    for k, v in sd_src.items():
        p = k.split('.')
        if p[0] == 'model' and p[1].isdigit() and int(p[1]) == n_src:
            p[1] = str(n_dst)
        remapped['.'.join(p)] = v
    sd_dst = model.state_dict()
    ok = {k: v for k, v in remapped.items() if k in sd_dst and v.shape == sd_dst[k].shape}
    model.load_state_dict(ok, strict=False)
    numel = lambda d: sum(v.numel() for k, v in d.items() if v.dtype.is_floating_point and 'running' not in k)
    info = dict(source=str(source), tensors_transferred=len(ok), tensors_total=len(sd_dst),
                params_transferred=numel(ok), params_total=numel(sd_dst))
    info['fraction_params_transferred'] = round(info['params_transferred'] / max(1, info['params_total']), 4)
    return info


def make_trainer(init='yolov8m.pt', log_path=None):
    """Trainer class that builds a GSWDetectionModel. A new run is initialised from `init`: either a YOLOv8m
    checkpoint (partial transfer, see transfer_weights) or a GSW-YOLO checkpoint of the same variant (full load).
    A resumed run loads its own last.pt."""

    class GSWTrainer(DetectionTrainer):
        INIT, LOG = init, log_path

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = GSWDetectionModel(cfg, nc=self.data['nc'], ch=self.data.get('channels', 3),
                                      verbose=verbose and RANK == -1)
            if hasattr(self, 'set_model_names_for_load'):
                model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
                return model
            info = transfer_weights(model, self.INIT)
            info['variant'] = model.yaml.get('variant')
            info['iou_loss'] = model.yaml.get('iou_loss')
            if self.LOG:
                os.makedirs(os.path.dirname(self.LOG), exist_ok=True)
                with open(self.LOG, 'w') as f:
                    json.dump(info, f, indent=2)
            print(f"Initialisation from {info['source']}: {info['tensors_transferred']}/{info['tensors_total']} "
                  f"tensors, {100 * info['fraction_params_transferred']:.1f}% of the parameters", flush=True)
            return model

    return GSWTrainer


def model_complexity(model, imgsz=640):
    """Parameters (M) and GFLOPs at the given input size."""
    from ultralytics.utils.torch_utils import get_flops
    m = model.model if hasattr(model, 'predictor') or hasattr(model, 'overrides') else model
    params = sum(p.numel() for p in m.parameters()) / 1e6
    try:
        gflops = float(get_flops(m, imgsz))
    except Exception:
        gflops = float('nan')
    return round(params, 3), round(gflops, 2)
