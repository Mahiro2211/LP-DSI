"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse

from engine.misc import dist_utils
from engine.core import YAMLConfig, yaml_utils
from engine.solver import TASKS

debug=False

if debug:
    import torch
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr

def main(args, ) -> None:
    """main
    """
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'


    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)

    # W&B opt-in: the solver's _wandb_init reads cfg.yaml_cfg['use_wandb'];
    # off by default and a soft dependency (training still runs if wandb
    # is not installed).
    cfg.yaml_cfg["use_wandb"] = bool(args.wandb)

    # Checkpoint opt-out: the solver skips all .pth weight files while
    # log.txt / eval stats / TensorBoard / W&B keep working. The stage-1
    # best snapshot the stage-2 restart needs is cached in RAM instead.
    # A yaml `save_ckpt: false` achieves the same without the flag.
    cfg.yaml_cfg["save_ckpt"] = cfg.yaml_cfg.get("save_ckpt", True) and not args.no_ckpt

    if args.resume or args.tuning:
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    print('cfg: ', cfg.__dict__)

    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    dist_utils.cleanup()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='resume from checkpoint')
    parser.add_argument('-t', '--tuning', type=str, help='tuning from checkpoint')
    parser.add_argument('-d', '--device', type=str, help='device',)
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('--use-amp', action='store_true', help='auto mixed precision training')
    parser.add_argument('--output-dir', type=str, help='output directoy')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry')
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--wandb', action='store_true', default=False,
        help='log metrics to Weights & Biases (requires `pip install wandb` and `wandb login`); '
             'resumed runs (-r) continue the same W&B run instead of starting a new one')
    parser.add_argument('--no-ckpt', action='store_true', default=False,
        help='skip weight checkpoints (last.pth / best_stg1.pth / best_stg2.pth); metrics are '
             'still written to log.txt / eval/ / TensorBoard / W&B. Such a run cannot be '
             'resumed after a crash; stage-2 restarts use an in-memory copy of the stage-1 '
             'best weights instead of best_stg1.pth')

    # priority 1
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')

    parser.add_argument('--local-rank', type=int, help='local rank id')
    args = parser.parse_args()

    main(args)
