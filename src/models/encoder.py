from typing import Optional

import torch
import torch.nn as nn

from src.models.attention import HierarchicalCrissCrossSummaryEncoder, mask_tokens
from src.models.config import HiBrainMJConfig
from src.models.patch_encoder import TFPatchEncoder
from src.models.positional_encoding import EEGSpatialTemporalPositionalEncoding


class EEGEncoder(nn.Module):
    """Patch encoder + EEG positional encoding + hierarchical criss-cross encoder.

    Input:  patches [B, C, T, patch_len]
    Output: latent grid [B, C, T, D]
    """

    def __init__(self, cfg: HiBrainMJConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_encoder = TFPatchEncoder(cfg.patch_len, cfg.embed_dim, cfg.dropout)
        self.pos_encoding = EEGSpatialTemporalPositionalEncoding(
            cfg.n_channels,
            cfg.embed_dim,
            cfg.spatial_pe_dim,
            cfg.temporal_pe_dim,
            cfg.montage_name,
            cfg.channel_names,
        )
        self.encoder = HierarchicalCrissCrossSummaryEncoder(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            patch_criss_cross_depth=cfg.patch_criss_cross_depth,
            summary_depth=cfg.summary_depth,
            dropout=cfg.dropout,
            drop_path=cfg.drop_path,
            use_gated_summary_injection=cfg.use_gated_summary_injection,
        )

    def positional_grid(self, t: int, device=None, dtype=None):
        return self.pos_encoding(t, device=device, dtype=dtype)

    def forward(self, patches: torch.Tensor, visible_mask: Optional[torch.Tensor] = None):
        if visible_mask is not None:
            patches = patches * visible_mask.unsqueeze(-1).to(patches.dtype)
        tokens = self.patch_encoder(patches)
        pe = self.positional_grid(tokens.shape[2], tokens.device, tokens.dtype)
        h0 = mask_tokens(tokens + pe.unsqueeze(0), visible_mask)
        return self.encoder(h0, visible_mask)
