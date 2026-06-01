import torch
import torch.nn as nn

from src.models.hibrainmj import EEGEncoder, HiBrainMJConfig


def _get(params, name, default):
    return getattr(params, name, default)


def _output_dim(params):
    return int(params.num_of_classes) if str(params.task_type).lower() == "multiclass" else 1


def _encoder_config(params):
    embed_dim = int(_get(params, "embed_dim", 768))
    spatial_dim = _get(params, "spatial_pe_dim", None)
    temporal_dim = _get(params, "temporal_pe_dim", None)
    return HiBrainMJConfig(
        n_channels=int(params.num_channels),
        patch_len=int(params.patch_size),
        embed_dim=embed_dim,
        num_heads=int(_get(params, "num_heads", 12)),
        mlp_ratio=float(_get(params, "mlp_ratio", 4.0)),
        patch_criss_cross_depth=int(_get(params, "patch_criss_cross_depth", 4)),
        summary_depth=int(_get(params, "summary_depth", 1)),
        dropout=float(_get(params, "dropout", 0.0)),
        drop_path=float(_get(params, "drop_path", 0.0)),
        use_gated_summary_injection=bool(_get(params, "use_gated_summary_injection", False)),
        use_geometry_bias=False,
        spatial_pe_dim=None if spatial_dim is None else int(spatial_dim),
        temporal_pe_dim=None if temporal_dim is None else int(temporal_dim),
        montage_name=str(_get(params, "montage_name", "standard_1020")),
        channel_names=_get(params, "channel_names", None),
    )


class HiBrainMJBackbone(nn.Module):
    """Finetune feature extractor.

    Input:  [B, C, T, patch_len] or [B, C, L]
    Output: [B, C*T, D]
    """

    def __init__(self, params):
        super().__init__()
        self.encoder = EEGEncoder(_encoder_config(params))
        self.embed_dim = self.encoder.cfg.embed_dim
        self.patch_len = self.encoder.cfg.patch_len

    def forward(self, x):
        if x.ndim == 3:
            t = x.shape[-1] // self.patch_len
            x = x[..., :t * self.patch_len].view(x.shape[0], x.shape[1], t, self.patch_len)
        z = self.encoder(x)
        return z.flatten(1, 2)


class DownstreamModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.backbone = HiBrainMJBackbone(params)
        if params.use_pretrained_weights and params.foundation_dir:
            self._load_pretrained_encoder(params.foundation_dir)

        embed_dim = int(self.backbone.embed_dim)
        flat_dim = int(params.num_channels) * int(params.num_patches) * embed_dim
        out_dim = _output_dim(params)

        if params.classifier == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(embed_dim, out_dim))
        elif params.classifier == "all_patch_reps_onelayer":
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, out_dim))
        elif params.classifier == "all_patch_reps_twolayer":
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, embed_dim),
                nn.ELU(),
                nn.Dropout(params.dropout),
                nn.Linear(embed_dim, out_dim),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, max(embed_dim * 4, 1)),
                nn.ELU(),
                nn.Dropout(params.dropout),
                nn.Linear(max(embed_dim * 4, 1), embed_dim),
                nn.ELU(),
                nn.Dropout(params.dropout),
                nn.Linear(embed_dim, out_dim),
            )

    def _load_pretrained_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("context_encoder", checkpoint.get("encoder", checkpoint))
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        state = {k[len("context_encoder."):] if k.startswith("context_encoder.") else k: v for k, v in state.items()}
        current = self.backbone.encoder.state_dict()
        state = {k: v for k, v in state.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
        msg = self.backbone.encoder.load_state_dict(state, strict=False)
        print(f"loaded HiBrainMJ context encoder from {checkpoint_path}")
        print(msg)

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(self.classifier[0], nn.AdaptiveAvgPool1d):
            feats = feats.transpose(1, 2)
        return self.classifier(feats).squeeze(-1)
