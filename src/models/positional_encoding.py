import math
from typing import Sequence

import torch
import torch.nn as nn


CHANNEL_ALIASES = {
    "FP1": "Fp1", "FP2": "Fp2", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}


def _channel_name(name: str) -> str:
    raw = str(name).replace("EEG", "").replace("-REF", "").replace("-LE", "").strip()
    key = raw.replace(" ", "").upper()
    return CHANNEL_ALIASES.get(key, raw[:1].upper() + raw[1:].lower())


def _fallback_standard_1020(names: Sequence[str]) -> torch.Tensor:
    xy = {
        "Fp1": (-0.45, 0.82), "Fp2": (0.45, 0.82),
        "F3": (-0.40, 0.45), "F4": (0.40, 0.45),
        "C3": (-0.45, 0.00), "C4": (0.45, 0.00),
        "P3": (-0.40, -0.45), "P4": (0.40, -0.45),
        "O1": (-0.35, -0.82), "O2": (0.35, -0.82),
        "F7": (-0.78, 0.40), "F8": (0.78, 0.40),
        "T7": (-0.92, 0.00), "T8": (0.92, 0.00),
        "P7": (-0.78, -0.40), "P8": (0.78, -0.40),
        "Fz": (0.00, 0.55), "Cz": (0.00, 0.00), "Pz": (0.00, -0.55),
    }
    missing = [ch for ch in names if ch not in xy]
    if missing:
        raise ValueError(f"Channels missing from fallback standard_1020 coordinates: {missing}")
    coords = []
    for ch in names:
        x, y = xy[ch]
        coords.append([x, y, math.sqrt(max(1.0 - x * x - y * y, 0.0))])
    return torch.tensor(coords, dtype=torch.float32)


def _standard_montage_coords(montage_name: str, channel_names: Sequence[str]) -> torch.Tensor:
    names = [_channel_name(ch) for ch in channel_names]
    try:
        import mne

        montage = mne.channels.make_standard_montage(montage_name)
        positions = montage.get_positions()["ch_pos"]
        missing = [ch for ch in names if ch not in positions]
        if missing:
            raise ValueError(f"Channels missing from MNE montage {montage_name}: {missing}")
        coords = torch.tensor([positions[ch] for ch in names], dtype=torch.float32)
    except ModuleNotFoundError:
        coords = _fallback_standard_1020(names)
    coords = coords - coords.mean(dim=0, keepdim=True)
    return coords / coords.norm(dim=1).max().clamp_min(1e-6)


def _sinusoidal(length: int, dim: int, device, dtype) -> torch.Tensor:
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    if dim <= 0:
        return pe.to(dtype)
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(dim, 1)))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


class EEGSpatialTemporalPositionalEncoding(nn.Module):
    """3D montage coordinate PE concatenated with sinusoidal temporal PE.

    Output for T patches: [C, T, D]
    """

    def __init__(
        self,
        n_channels: int,
        embed_dim: int,
        spatial_pe_dim: int,
        temporal_pe_dim: int,
        montage_name: str,
        channel_names: Sequence[str],
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.embed_dim = int(embed_dim)
        self.spatial_pe_dim = int(spatial_pe_dim)
        self.temporal_pe_dim = int(temporal_pe_dim)
        coords = _standard_montage_coords(montage_name, list(channel_names)[:n_channels])
        self.register_buffer("coords", coords, persistent=True)
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, self.spatial_pe_dim),
            nn.GELU(),
            nn.Linear(self.spatial_pe_dim, self.spatial_pe_dim),
        )

    def forward(self, num_time_patches: int, device=None, dtype=None) -> torch.Tensor:
        device = device or self.coords.device
        dtype = dtype or self.coords.dtype
        spatial = self.spatial_proj(self.coords.to(device=device, dtype=dtype))
        temporal = _sinusoidal(int(num_time_patches), self.temporal_pe_dim, device, dtype)
        spatial = spatial[:, None, :].expand(self.n_channels, int(num_time_patches), -1)
        temporal = temporal[None, :, :].expand(self.n_channels, int(num_time_patches), -1)
        return torch.cat([spatial, temporal], dim=-1)
