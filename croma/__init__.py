"""Vendored inference-only copy of CROMA (NeurIPS 2023, arXiv 2311.00566).

Only `use_croma.py` + einops are required for inference, per the official repo:
https://github.com/antofuller/croma
The file is kept verbatim so it stays diffable against upstream.
"""
from .use_croma import PretrainedCROMA, ViT, get_2dalibi

__all__ = ["PretrainedCROMA", "ViT", "get_2dalibi"]
