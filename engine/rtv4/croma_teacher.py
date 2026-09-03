"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
CROMA teacher: SAR-domain foundation model (NeurIPS 2023, arXiv 2311.00566)
as a frozen distillation teacher (replacing the natural-image DINOv3 teacher
of the RT-DETRv4 baseline). Code vendored under `croma/` (official
`use_croma.py`, unmodified).
"""

import itertools
import logging
import math

import torch
import torch.nn as nn

from ..core import register
from croma.use_croma import PretrainedCROMA

_logger = logging.getLogger(__name__)


def build_2dalibi_rect(num_heads, h, w):
    """Generalization of croma.use_croma.get_2dalibi to rectangular grids.

    Identical to the official function when h == w (same slope math and
    row-major point ordering), so pretrained behaviour is unchanged.
    """
    points = list(itertools.product(range(h), range(w)))

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                :n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, h * w, h * w)


@register()
class CROMATeacherModel(nn.Module):
    """Frozen CROMA SAR encoder as distillation teacher.

    Output contract (same as the DINOv3 teacher of the RT-DETRv4 baseline): a
    detached (B, teacher_dim, H/32, W/32) feature map whose grid matches the
    student's F5 (stride-32) feature 1:1 for any input size divisible by 32.

    - Input: (B, 3, H, W) in [0, 1] (SAR grayscale replicated to 3 channels on
      disk). CROMA's SAR encoder takes 2 channels (VV/VH); the mean channel is
      broadcast to both slots since the detection data is single-polarization.
    - Normalization: CROMA pretraining clips SAR per-channel to 8-bit values
      then divides by 255, i.e. the encoder consumes [0, 1] inputs -- exactly
      what the dataloader already produces, so no extra normalization.
    - Alignment: AvgPool(input_downsample=4) + ViT patch 8 -> effective stride
      32 (640 -> 160 -> 20x20 grid), same granularity as the baseline's
      DINOv3 teacher (AvgPool 2 + patch 16), and 160x160 stays close to
      CROMA's 120x120 pretraining resolution.
    """

    def __init__(self,
                 croma_weights_path: str = 'pretrain/CROMA_base.pt',
                 croma_size: str = 'base',
                 modality: str = 'SAR',
                 patch_size: int = 8,
                 input_downsample: int = 4):
        super().__init__()
        if modality != 'SAR':
            raise ValueError(
                f"CROMATeacherModel only supports modality='SAR' (detection data has no optical input), got '{modality}'.")

        self.croma_weights_path = croma_weights_path
        self.croma_size = croma_size
        self.patch_size = patch_size
        self.input_downsample = input_downsample

        _logger.info(f"[Teacher Model] Attempting to load CROMA teacher (size={croma_size})...")
        _logger.info(f"[Teacher Model] CROMA weights path: {croma_weights_path}")

        try:
            # Nominal pretraining-resolution image size for the initial ALiBi;
            # the bias is recomputed dynamically for any other grid in forward.
            nominal_resolution = 640 // input_downsample
            self.model = PretrainedCROMA(
                pretrained_path=croma_weights_path,
                size=croma_size,
                modality='SAR',
                image_resolution=nominal_resolution,
            )
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            if self.model.patch_size != patch_size:
                _logger.warning(
                    f"[Teacher Model] Configured patch_size={patch_size} but CROMA uses "
                    f"patch_size={self.model.patch_size}; using the model's value.")
            self.patch_size = self.model.patch_size

            self.teacher_feature_dim = self.model.encoder_dim
            _logger.info(f"[Teacher Model] Successfully loaded and froze CROMA teacher.")

        except Exception as e:
            _logger.error(f"[Teacher Model] Failed to load CROMA: {e}")
            raise

        self.avgpool = nn.AvgPool2d(kernel_size=input_downsample, stride=input_downsample)
        # PretrainedCROMA.attn_bias is a plain tensor attribute (not a buffer),
        # so it neither follows .to(device) nor adapts to varying input grids.
        # Manage it here, cached per (grid, device).
        self._attn_bias_cache = {}

        _logger.info(f"[Teacher Model] CROMA initialized. Feature dimension: {self.teacher_feature_dim}.")
        _logger.info(
            f"[Teacher Model] Effective teacher stride: {input_downsample} * {self.patch_size} = "
            f"{input_downsample * self.patch_size}, matching the student's highest-level (F5) features 1:1.")

    def _ensure_attn_bias(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._attn_bias_cache:
            self._attn_bias_cache[key] = build_2dalibi_rect(
                self.model.num_heads, h, w).to(device)
        # PretrainedCROMA.forward applies .to(device) on this tensor itself,
        # which is a no-op once it already lives on the right device.
        self.model.attn_bias = self._attn_bias_cache[key]

    def forward(self, images: torch.Tensor):
        B, _, H, W = images.shape
        effective_stride = self.input_downsample * self.patch_size
        if H % effective_stride or W % effective_stride:
            _logger.error(
                f"[Teacher Model] Input size ({H}x{W}) must be divisible by the effective stride "
                f"{effective_stride} (input_downsample * patch_size).")
            raise ValueError(
                f"Input size ({H}x{W}) not divisible by teacher effective stride {effective_stride}.")

        sar = images.mean(dim=1, keepdim=True).expand(-1, self.model.s1_channels, -1, -1)
        sar = self.avgpool(sar)  # (B, 2, H/downsample, W/downsample)

        h = sar.shape[-2] // self.patch_size
        w = sar.shape[-1] // self.patch_size

        with torch.no_grad():
            self._ensure_attn_bias(h, w, sar.device)
            croma_output_dict = self.model(SAR_images=sar)
            patch_tokens = croma_output_dict["SAR_encodings"]  # (B, h*w, C)

            if patch_tokens.dim() != 3 or patch_tokens.shape[1] != h * w:
                _logger.error(
                    f"[Teacher Model] Unexpected CROMA output shape {tuple(patch_tokens.shape)} "
                    f"for grid {h}x{w}.")
                raise ValueError("CROMA SAR_encodings shape does not match the expected patch grid.")

            teacher_feature_map = patch_tokens.permute(0, 2, 1).reshape(B, self.teacher_feature_dim, h, w)

            return teacher_feature_map.detach()
