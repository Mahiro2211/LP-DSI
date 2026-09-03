"""
Training-only distillation projector with configurable capacity.

The default projector is a per-token Linear map (hidden_dim -> teacher_dim)
whose only job is dimension alignment for feature distillation. MLPProjector
adds a controlled Linear-GELU-Linear capacity arm.

All variants live exclusively in the training-time distillation branch of
HybridEncoder (guarded by `self.training`): inference cost and deployed
checkpoints are unchanged.
"""

import torch.nn as nn


class MLPProjector(nn.Module):
    """Per-token Linear-GELU-Linear adapter: the capacity control arm."""

    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = out_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
