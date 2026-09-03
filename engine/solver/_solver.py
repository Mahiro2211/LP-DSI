"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    if m is None:
        return None
    return m.to(device)


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg

    def _setup(self):
        """Avoid instantiating unnecessary classes"""
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = cfg.model

        # Setup teacher model
        self.teacher_model = to(cfg.teacher_model, device)

        # NOTE: Must load_tuning_state before EMA instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        self.model = dist_utils.warp_model(
            self.model.to(device), sync_bn=cfg.sync_bn, find_unused_parameters=cfg.find_unused_parameters
        )

        self.criterion = self.to(cfg.criterion, device)
        self.postprocessor = self.to(cfg.postprocessor, device)

        self.ema = self.to(cfg.ema, device)
        self.scaler = cfg.scaler

        self.device = device
        self.last_epoch = self.cfg.last_epoch

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cfg.writer

        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text('config', '{:s}'.format(cfg.__repr__()), 0)

    def cleanup(self):
        if self.writer:
            atexit.register(self.writer.close)

    def train(self):
        self._setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader, shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # NOTE: Instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self):
        self._setup()

        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, module, device):
        return module.to(device) if hasattr(module, 'to') else module

    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def load_state_dict(self, state):
        """Load state dict, train/eval"""
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        if 'wandb_run_id' in state:
            self._wandb_run_id = state['wandb_run_id']
            print(f"Load wandb_run_id: {state['wandb_run_id']}")

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                if k == 'ema':
                    model = getattr(self, 'model', None)
                    if model is not None:
                        ema = dist_utils.de_parallel(v)
                        model_state_dict = remove_module_prefix(model.state_dict())
                        ema.load_state_dict({'module': model_state_dict})
                        print(f'Load {k}.state_dict from model.state_dict')
                else:
                    print(f'Not load {k}.state_dict')

    def load_resume_state(self, path: str):
        """Load resume"""
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        # state['model'] = remove_module_prefix(state['model'])
        self.load_state_dict(state)

    def load_tuning_state(self, path: str):
        """Load model for tuning and adjust mismatched head parameters"""
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        module = dist_utils.de_parallel(self.model)

        # Load the appropriate state dict
        if 'ema' in state:
            pretrain_state_dict = state['ema']['module']
        else:
            pretrain_state_dict = state['model']

        # Adjust head parameters between datasets
        try:
            adjusted_state_dict = self._adjust_head_parameters(module.state_dict(), pretrain_state_dict)
            stat, infos = self._matched_state(module.state_dict(), adjusted_state_dict)
        except Exception:
            stat, infos = self._matched_state(module.state_dict(), pretrain_state_dict)

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        missed_list = []
        unmatched_list = []
        matched_state = {}
        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def _adjust_head_parameters(self, cur_state_dict, pretrain_state_dict):
        """Adjust head parameters between datasets."""
        # List of parameters to adjust
        if pretrain_state_dict['decoder.denoising_class_embed.weight'].size() != \
                cur_state_dict['decoder.denoising_class_embed.weight'].size():
            del pretrain_state_dict['decoder.denoising_class_embed.weight']

        head_param_names = [
            'decoder.enc_score_head.weight',
            'decoder.enc_score_head.bias'
        ]
        for i in range(8):
            head_param_names.append(f'decoder.dec_score_head.{i}.weight')
            head_param_names.append(f'decoder.dec_score_head.{i}.bias')

        for param_name in head_param_names:
            if param_name in cur_state_dict and param_name in pretrain_state_dict:
                cur_tensor = cur_state_dict[param_name]
                pretrain_tensor = pretrain_state_dict[param_name]
                if pretrain_tensor.size() == cur_tensor.size():
                    pretrain_state_dict[param_name] = pretrain_tensor
                else:
                    print(f"Skip parameter '{param_name}': size mismatch "
                          f"{tuple(pretrain_tensor.size())} vs {tuple(cur_tensor.size())}.")

        return pretrain_state_dict

    def fit(self):
        raise NotImplementedError('')

    def val(self):
        raise NotImplementedError('')
