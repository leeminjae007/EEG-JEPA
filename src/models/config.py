from dataclasses import dataclass, fields
from typing import Dict, Optional, Sequence


DEFAULT_19_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]


@dataclass
class HiBrainMJConfig:
    n_channels: int = 19
    patch_len: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_criss_cross_depth: int = 4
    summary_depth: int = 1
    dropout: float = 0.0
    drop_path: float = 0.1
    use_gated_summary_injection: bool = False
    use_geometry_bias: bool = False
    spatial_pe_dim: Optional[int] = None
    temporal_pe_dim: Optional[int] = None
    montage_name: str = "standard_1020"
    channel_names: Optional[Sequence[str]] = None
    target_scale: Sequence[float] = (0.15, 0.2)
    context_scale: Sequence[float] = (0.85, 1.0)
    aspect_ratio: Sequence[float] = (0.75, 1.5)
    num_target_blocks: int = 4
    num_context_blocks: int = 1
    ema_momentum: float = 0.996
    pred_depth: int = 6
    pred_num_heads: Optional[int] = None
    pred_mlp_ratio: float = 4.0

    def __post_init__(self):
        if self.channel_names is None:
            self.channel_names = DEFAULT_19_CHANNELS[: self.n_channels]
        self.spatial_pe_dim = self.embed_dim // 2 if self.spatial_pe_dim is None else int(self.spatial_pe_dim)
        self.temporal_pe_dim = self.embed_dim - self.spatial_pe_dim if self.temporal_pe_dim is None else int(self.temporal_pe_dim)
        if self.spatial_pe_dim + self.temporal_pe_dim != self.embed_dim:
            self.temporal_pe_dim = self.embed_dim - self.spatial_pe_dim

    @classmethod
    def from_dict(cls, cfg: Dict):
        valid = {f.name for f in fields(cls)}
        data = {}
        data.update(cfg.get("model", {}))
        data.update(cfg.get("hibrainmj", {}))
        data.update({k: v for k, v in cfg.items() if k in valid})
        return cls(**{k: v for k, v in data.items() if k in valid})
