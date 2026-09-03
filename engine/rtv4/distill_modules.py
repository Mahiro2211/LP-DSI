"""
Distillation alignment losses for RT-DETRv4 (SAR ship detection).

Low-pass alignment rationale: the VFM teacher and the SAR student live in
different modalities.  What transfers across modalities is the low-frequency
semantic structure of the feature map, while the high-frequency content is
dominated by modality-specific texture (for SAR, the pixel-level multiplicative
speckle that survives as token-level detail).  Aligning the full-spectrum
features therefore wastes gradient on detail the teacher is not trustworthy
about.  We align only the low-pass component.

Concretely the low-pass operator P is a 2x2 average pooling, which is exactly
the Haar wavelet LL sub-band (LL = 2 * avg_pool2d(x, 2)); cosine similarity is
invariant to the constant factor, so low-pass alignment is the minimal,
hyperparameter-free form of frequency-decoupled distillation.  All operators
are parameter-free and used only inside the training loss, so inference cost
is unchanged.
"""

import torch.nn.functional as F


def cosine_alignment_loss(student, teacher):
    """Mean (1 - cosine) over spatial locations between two feature maps.

    Args:
        student, teacher: (B, C, H, W).
    """
    s = F.normalize(student.flatten(2).permute(0, 2, 1), p=2, dim=-1)
    t = F.normalize(teacher.flatten(2).permute(0, 2, 1), p=2, dim=-1)
    return (1 - (s * t).sum(dim=-1)).mean()


def low_pass_alignment_loss(student, teacher, stride=2):
    """Align only the low-frequency (smoothed) component of the features.

    Both maps are average-pooled by `stride` before the per-position cosine
    loss, which suppresses speckle- and texture-dominated high-frequency
    responses that do not transfer across modalities.  Average pooling is the
    Haar LL sub-band, so this is the minimal form of frequency-decoupled
    distillation with zero extra hyperparameters; odd spatial sizes are simply
    floor-cropped by the pooling, no padding is needed.

    Args:
        student, teacher: (B, C, H, W) feature maps already spatially aligned.
        stride: pooling stride (2 keeps the standard dyadic low-pass band).
    """
    s = F.avg_pool2d(student, stride)
    t = F.avg_pool2d(teacher, stride)
    return cosine_alignment_loss(s, t)
