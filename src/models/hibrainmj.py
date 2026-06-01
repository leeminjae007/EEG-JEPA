import copy
from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.masks.random import BlockMaskGenerator, EEGMaskBatch
from src.models.attention import SummaryAttentionBlock, mask_tokens
from src.models.config import HiBrainMJConfig
from src.models.encoder import EEGEncoder


class HiBrainMJPredictor(nn.Module):
    """Predict target EMA latents from context latents and target positions.

    context: [B, C, T, D]
    pos_grid: [C, T, D]
    output:  [B, C, T, D], non-zero at target positions
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_proj = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            SummaryAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context, pos_grid, context_mask, target_mask):
        b, c, t, d = context.shape
        n = c * t
        ctx = self.context_proj(context.reshape(b, n, d))
        pos = self.pos_proj(pos_grid.reshape(1, n, d)).expand(b, n, d)
        target = target_mask.reshape(b, n).bool()
        active = context_mask.reshape(b, n).bool() | target
        x = torch.where(target.unsqueeze(-1), self.mask_token + pos, ctx)
        x = mask_tokens(x, active)
        for block in self.blocks:
            x = block(x, active)
        x = self.out_proj(self.norm(x)).view(b, c, t, d)
        return mask_tokens(x, target_mask)


class HiBrainMJ(nn.Module):
    """HiBrainMJ-style EEG representation learner.

    Raw input: [B, C, L]
    Patch input: [B, C, T, patch_len]
    """

    def __init__(self, config: Optional[HiBrainMJConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = HiBrainMJConfig(**kwargs)
        elif isinstance(config, dict):
            config = HiBrainMJConfig.from_dict({**config, **kwargs})
        elif kwargs:
            data = asdict(config)
            data.update(kwargs)
            config = HiBrainMJConfig(**data)
        self.cfg = config

        self.context_encoder = EEGEncoder(config)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = HiBrainMJPredictor(
            embed_dim=config.embed_dim,
            depth=config.pred_depth,
            num_heads=config.pred_num_heads or config.num_heads,
            mlp_ratio=config.pred_mlp_ratio,
            dropout=config.dropout,
            drop_path=config.drop_path,
        )
        self.mask_generator = BlockMaskGenerator(
            input_size=None,
            target_scale=config.target_scale,
            context_scale=config.context_scale,
            aspect_ratio=config.aspect_ratio,
            num_target_blocks=config.num_target_blocks,
            num_context_blocks=config.num_context_blocks,
        )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x
        b, c, l = x.shape
        t = l // self.cfg.patch_len
        x = x[:, :, :t * self.cfg.patch_len]
        return x.view(b, c, t, self.cfg.patch_len)

    @staticmethod
    def _indices_to_mask(indices, b, c, t, device):
        mask = torch.zeros(b, c * t, dtype=torch.bool, device=device)
        for row in range(b):
            ids = indices[row][indices[row].ge(0)].long()
            mask[row, ids] = True
        return mask.view(b, c, t)

    def _resolve_masks(self, masks, b, c, t, device) -> EEGMaskBatch:
        if masks is None:
            return self.mask_generator(b, input_size=(c, t), device=device)
        if isinstance(masks, EEGMaskBatch):
            return masks.to(device)
        if isinstance(masks, dict):
            context = masks["context_mask"].to(device)
            target = masks["target_mask"].to(device)
        else:
            context, target = masks
            context = context[0] if isinstance(context, (list, tuple)) else context
            target = target[0] if isinstance(target, (list, tuple)) else target
            context, target = context.to(device), target.to(device)
        context = self._indices_to_mask(context, b, c, t, device) if context.ndim == 2 and context.dtype != torch.bool else context.bool()
        target = self._indices_to_mask(target, b, c, t, device) if target.ndim == 2 and target.dtype != torch.bool else target.bool()
        return EEGMaskBatch(
            context_mask=context,
            target_mask=target,
            context_indices=BlockMaskGenerator._stack_indices(list(context.cpu())),
            target_indices=BlockMaskGenerator._stack_indices(list(target.cpu())),
        ).to(device)

    def forward(self, x, masks=None):
        patches = self.patchify(x)
        b, c, t, _ = patches.shape
        masks = self._resolve_masks(masks, b, c, t, patches.device)
        target_mask = masks.target_mask.bool()
        context_mask = masks.context_mask.bool() & ~target_mask

        context = self.context_encoder(patches, context_mask)
        with torch.no_grad():
            target = self.target_encoder(patches, target_mask)

        pos = self.context_encoder.positional_grid(t, patches.device, context.dtype)
        pred = self.predictor(context, pos, context_mask, target_mask)
        loss = F.mse_loss(pred[target_mask], target[target_mask].detach())
        return {
            "loss": loss,
            "pred": pred,
            "target": target.detach(),
            "context": context,
            "context_mask": context_mask,
            "target_mask": target_mask,
            "masks": masks,
        }

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None):
        m = self.cfg.ema_momentum if momentum is None else float(momentum)
        for target, context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target.data.mul_(m).add_(context.data, alpha=1.0 - m)

    @torch.no_grad()
    def reset_target_encoder(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False


def build_hibrainmj(config: Dict) -> HiBrainMJ:
    return HiBrainMJ(HiBrainMJConfig.from_dict(config))
