from typing import Optional

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep) * mask.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def safe_padding_mask(visible: Optional[torch.Tensor]):
    if visible is None:
        return None
    pad = ~visible.bool()
    empty = pad.all(dim=1)
    if empty.any():
        pad = pad.clone()
        pad[empty, 0] = False
    return pad


def mask_tokens(x: torch.Tensor, visible: Optional[torch.Tensor]):
    return x if visible is None else x * visible.unsqueeze(-1).to(x.dtype)


class CrissCrossBlock(nn.Module):
    """Temporal then spatial self-attention over an EEG [C, T] grid.

    Input/output: x [B, C, T, D], visible_mask [B, C, T]
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.spatial_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Linear(2 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    @staticmethod
    def _attend(attn, x, visible):
        out = attn(x, x, x, key_padding_mask=safe_padding_mask(visible), need_weights=False)[0]
        return mask_tokens(out, visible)

    def forward(self, x, visible_mask: Optional[torch.Tensor] = None):
        b, c, t, d = x.shape
        vt = None if visible_mask is None else visible_mask.reshape(b * c, t)
        vs = None if visible_mask is None else visible_mask.permute(0, 2, 1).reshape(b * t, c)

        xt = self.temporal_norm(x).reshape(b * c, t, d)
        xt = self._attend(self.temporal_attn, xt, vt).view(b, c, t, d)

        xs = self.spatial_norm(x).permute(0, 2, 1, 3).reshape(b * t, c, d)
        xs = self._attend(self.spatial_attn, xs, vs).view(b, t, c, d).permute(0, 2, 1, 3)

        x = self.norm1(x + self.drop_path(self.fuse(torch.cat([xt, xs], dim=-1))))
        x = mask_tokens(x, visible_mask)
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return mask_tokens(x, visible_mask)


class SummaryAttentionBlock(nn.Module):
    """Transformer block for channel or time summaries.

    Input/output: x [B, N, D], visible [B, N]
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x, visible: Optional[torch.Tensor] = None):
        h = self.attn_norm(x)
        y = self.attn(h, h, h, key_padding_mask=safe_padding_mask(visible), need_weights=False)[0]
        y = mask_tokens(y, visible)
        x = self.norm1(x + self.drop_path(y))
        x = mask_tokens(x, visible)
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return mask_tokens(x, visible)


def masked_mean(x, mask, dim):
    if mask is None:
        return x.mean(dim=dim)
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1.0)


class HierarchicalCrissCrossSummaryEncoder(nn.Module):
    """Patch-level criss-cross attention + summary attention + final injection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        patch_criss_cross_depth: int = 4,
        summary_depth: int = 1,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_gated_summary_injection: bool = False,
    ):
        super().__init__()
        dpr = torch.linspace(0, drop_path, patch_criss_cross_depth + summary_depth).tolist()
        self.patch_blocks = nn.ModuleList([
            CrissCrossBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(patch_criss_cross_depth)
        ])
        offset = patch_criss_cross_depth
        self.channel_summary_blocks = nn.ModuleList([
            SummaryAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[offset + i])
            for i in range(summary_depth)
        ])
        self.time_summary_blocks = nn.ModuleList([
            SummaryAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[offset + i])
            for i in range(summary_depth)
        ])
        self.use_gated_summary_injection = bool(use_gated_summary_injection)
        if self.use_gated_summary_injection:
            self.channel_gate = nn.Linear(2 * embed_dim, embed_dim)
            self.time_gate = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h0, visible_mask: Optional[torch.Tensor] = None):
        p = mask_tokens(h0, visible_mask)
        for block in self.patch_blocks:
            p = block(p, visible_mask)

        channel_visible = None if visible_mask is None else visible_mask.any(dim=2)
        time_visible = None if visible_mask is None else visible_mask.any(dim=1)
        gc = masked_mean(p, visible_mask, dim=2)
        gt = masked_mean(p, visible_mask, dim=1)

        for block in self.channel_summary_blocks:
            gc = block(gc, channel_visible)
        for block in self.time_summary_blocks:
            gt = block(gt, time_visible)

        gc = gc[:, :, None, :].expand_as(p)
        gt = gt[:, None, :, :].expand_as(p)
        if self.use_gated_summary_injection:
            alpha = torch.sigmoid(self.channel_gate(torch.cat([p, gc], dim=-1)))
            beta = torch.sigmoid(self.time_gate(torch.cat([p, gt], dim=-1)))
            z = p + alpha * gc + beta * gt
        else:
            z = p + gc + gt
        return mask_tokens(self.norm(z), visible_mask)
