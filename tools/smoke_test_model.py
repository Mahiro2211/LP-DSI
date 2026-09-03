"""
Model-level smoke test: build model + criterion + teacher from a config file,
run one training step (forward + loss + backward) and one inference step,
asserting outputs are finite and expected loss keys appear.

Complements tools/dataset/smoke_test_hrsid_ssdd.py (data-only) by covering the
model construction path, including the CROMA distillation teacher and the
LP-DSI low-pass alignment loss (distill_mode: lp).

Usage:
    python tools/smoke_test_model.py configs/rtv4_s_hrsid_lp.yml [more.yml ...]
    python tools/smoke_test_model.py --all
    python tools/smoke_test_model.py --ops    # pure-function checks only
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core import YAMLConfig
from engine.rtv4.croma_teacher import CROMATeacherModel
from engine.rtv4.distill_modules import cosine_alignment_loss, low_pass_alignment_loss

TEACHER_TYPES = {
    'CROMATeacherModel': CROMATeacherModel,
}

# The paper's configs: three lp runs (LP-DSI) and the CROMA cosine baselines
# they inherit from (teacher / loss ablation reference).
PAPER_CONFIGS = [
    'configs/rtv4_hgnetv2_s_hrsid_croma.yml',
    'configs/rtv4_hgnetv2_s_ship_croma.yml',
    'configs/rtv4_hgnetv2_s_ssdd_croma.yml',
    'configs/rtv4_s_hrsid_lp.yml',
    'configs/rtv4_s_ship_lp.yml',
    'configs/rtv4_s_ssdd_lp.yml',
]

IMG = 320        # keeps teacher and student F5 grids aligned at stride 32:
                 # CROMA 320/4/8 = 10, student 320/32 = 10
NUM_BOXES = 3


def make_targets(device, num_classes=1):
    boxes = torch.tensor([[0.30, 0.30, 0.06, 0.05],
                          [0.60, 0.55, 0.04, 0.06],
                          [0.45, 0.75, 0.10, 0.03]], device=device)
    labels = torch.zeros(NUM_BOXES, dtype=torch.int64, device=device) % max(num_classes, 1)
    return [{'labels': labels, 'boxes': boxes}, {'labels': labels[:2], 'boxes': boxes[:2]}]


def smoke_ops():
    """Pure-function checks for the distillation alignment losses."""
    torch.manual_seed(0)

    # --- cosine alignment ---
    s = torch.randn(2, 8, 20, 20)
    t = torch.randn(2, 8, 20, 20)
    val = cosine_alignment_loss(s, t)
    assert val.ndim == 0 and math.isfinite(val.item()) and 0.0 <= val.item() <= 2.0
    assert cosine_alignment_loss(t, t).item() < 1e-6, 'self-alignment must be ~0'

    # --- low-pass alignment: pooled band, scale invariance ---
    lp = low_pass_alignment_loss(s, t)
    manual = cosine_alignment_loss(F.avg_pool2d(s, 2), F.avg_pool2d(t, 2))
    assert torch.allclose(lp, manual), 'low-pass loss must equal cosine on the pooled band'
    assert low_pass_alignment_loss(t, t).item() < 1e-6
    # cosine is scale-invariant, so a constant factor on the pooled map (the
    # Haar LL convention is 2x avg_pool) must not change the loss
    assert torch.allclose(low_pass_alignment_loss(s, t),
                          cosine_alignment_loss(2 * F.avg_pool2d(s, 2), 2 * F.avg_pool2d(t, 2)))

    # --- odd token grids (multiscale training: 608/672 input -> 19/21 tokens) ---
    # avg_pool floor-crops odd sizes; no padding is needed anywhere.
    s19, t19 = torch.randn(2, 8, 19, 19), torch.randn(2, 8, 19, 19)
    assert F.avg_pool2d(s19, 2).shape[-2:] == (9, 9), 'odd grids must floor-crop'
    vals = (low_pass_alignment_loss(s19, t19),
            cosine_alignment_loss(s19, t19))
    assert all(math.isfinite(v.item()) for v in vals)
    s21, t21 = torch.randn(2, 8, 21, 17), torch.randn(2, 8, 21, 17)
    assert math.isfinite(low_pass_alignment_loss(s21, t21).item())

    print('ops OK: cosine / low-pass alignment / odd grids')


def smoke_one(config_path, device):
    print(f'\n=== {config_path} ===')
    cfg = YAMLConfig(config_path)

    model, criterion, postprocessor = cfg.model, cfg.criterion, cfg.postprocessor
    model.to(device).train()
    criterion.to(device)

    teacher = None
    tcfg = cfg.yaml_cfg.get('teacher_model')
    if tcfg is not None:
        ttype = tcfg.get('type')
        if TEACHER_TYPES.get(ttype) is None:
            raise ValueError(f'unsupported teacher_model type: {ttype}')
        teacher_cls = TEACHER_TYPES[ttype]
        teacher = teacher_cls(**{k: v for k, v in tcfg.items() if k != 'type'})
        teacher.to(device).eval()

    images = torch.rand(2, 3, IMG, IMG, device=device)
    targets = make_targets(device, num_classes=cfg.yaml_cfg.get('num_classes', 1))

    teacher_map = teacher(images) if teacher is not None else None

    print(f'pre-forward: model.training={model.training}, encoder.training={model.encoder.training}')
    outputs = model(images, targets, teacher_encoder_output=teacher_map)
    losses = criterion(outputs, targets)
    total = sum(l for l in losses.values() if torch.is_tensor(l))
    total.backward()

    bad = {k: v.item() for k, v in losses.items() if torch.is_tensor(v) and not math.isfinite(v.item())}
    assert not bad, f'non-finite losses: {bad}'

    keys = set(losses.keys())
    losses_list = cfg.yaml_cfg.get('RTv4Criterion', {}).get('losses', [])
    assert any('loss_mal' in k for k in keys), 'classification loss missing'
    assert any('loss_bbox' in k for k in keys), 'box loss missing'
    assert any('loss_fgl' in k for k in keys), 'FGL loss missing'
    if 'distill' in losses_list:
        assert any('loss_distill' in k for k in keys), 'distill loss missing'
        assert losses['loss_distill'].item() > 0, 'distill loss must be non-zero with a teacher'

    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, 'no gradient flowed to the model'

    # inference path — input must match eval_spatial_size because the encoder
    # caches its positional embedding buffer for deployment
    model.eval()
    eval_size = cfg.yaml_cfg.get('eval_spatial_size', [IMG, IMG])
    images_eval = torch.rand(2, 3, eval_size[0], eval_size[1], device=device)
    with torch.no_grad():
        out = model(images_eval)
        results = postprocessor(out, torch.tensor([eval_size] * 2, device=device))
    assert all(torch.isfinite(r['scores']).all().item() for r in results)

    print(f'train total loss = {total.item():.3f} ({len(losses)} terms), '
          f'grad sum = {grad_norm:.1f}, eval boxes = {results[0]["boxes"].shape[0]}')
    print('OK')

    del model, criterion, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='*')
    parser.add_argument('--all', action='store_true', help='run every paper config')
    parser.add_argument('--ops', action='store_true', help='pure-function checks only (no configs)')
    args = parser.parse_args()

    if args.ops:
        smoke_ops()
        sys.exit(0)

    configs = PAPER_CONFIGS if args.all else args.configs
    assert configs, 'provide config paths or --all'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    smoke_ops()
    for c in configs:
        smoke_one(c, device)
    print('\nALL SMOKE TESTS PASSED')
