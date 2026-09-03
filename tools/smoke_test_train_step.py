"""
Training-loop smoke test: drive engine.solver.det_engine.train_one_epoch for a
couple of synthetic batches (forward + loss + backward + optimizer step),
covering the paths that model-level tests cannot: AMP autocast boundaries and
the GAM gradient probe.

Usage:
    python tools/smoke_test_train_step.py configs/rtv4_s_hrsid_lp.yml [more.yml ...]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.cuda.amp.grad_scaler import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.core import YAMLConfig
from engine.solver.det_engine import train_one_epoch

IMG = 320
BATCH = 2


def make_batch(device, img):
    samples = torch.rand(BATCH, 3, img, img)
    boxes = torch.tensor([[0.30, 0.30, 0.06, 0.05],
                          [0.60, 0.55, 0.04, 0.06],
                          [0.45, 0.75, 0.10, 0.03]], device=device)
    labels = torch.zeros(3, dtype=torch.int64, device=device)
    targets = [{'labels': labels, 'boxes': boxes},
               {'labels': labels[:2], 'boxes': boxes[:2]}]
    return samples, targets


def smoke_train_step(config_path, device, use_amp, img=IMG):
    amp_tag = 'amp' if use_amp else 'fp32'
    print(f'\n=== {config_path} [{amp_tag}, img={img}] ===')
    cfg = YAMLConfig(config_path)

    model, criterion = cfg.model, cfg.criterion
    model.to(device).train()
    criterion.to(device).train()

    teacher = None
    tcfg = cfg.yaml_cfg.get('teacher_model')
    if tcfg is not None:
        from engine.rtv4.croma_teacher import CROMATeacherModel
        cls = {'CROMATeacherModel': CROMATeacherModel}[tcfg['type']]
        teacher = cls(**{k: v for k, v in tcfg.items() if k != 'type'}).to(device).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    data = [make_batch('cpu', img) for _ in range(2)]  # samples moved to device in the loop

    stats, grad_pcts = train_one_epoch(
        False, None, model, criterion, data, optimizer, device, epoch=0,
        max_norm=0.1, print_freq=1, ema=None, scaler=scaler,
        lr_warmup_scheduler=None, writer=None,
        teacher_model=teacher)

    assert 'loss' in stats and torch.isfinite(torch.tensor(stats['loss'])), stats
    has_gam = bool(criterion.distill_adaptive_params and criterion.distill_adaptive_params.get('enabled'))
    assert bool(grad_pcts) == has_gam, f'GAM probe mismatch: {len(grad_pcts)} samples, enabled={has_gam}'
    print(f"mean loss = {stats['loss']:.3f}, gam probes = {len(grad_pcts)} -> OK")

    del model, criterion, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--img', type=int, default=IMG,
                        help='square input size; 608 yields an odd 19x19 token grid '
                             '(the multiscale training case)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    for c in args.configs:
        smoke_train_step(c, device, use_amp=bool(device.type == 'cuda'), img=args.img)
    print('\nTRAIN-STEP SMOKE TESTS PASSED')
