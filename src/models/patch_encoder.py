import torch
import torch.nn as nn


def _groups(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class TFPatchEncoder(nn.Module):
    """CBraMod-style time-frequency patch encoder.

    Input:  x [B, C, T, patch_len]
    Output: tokens [B, C, T, D]
    """

    def __init__(self, patch_len: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.patch_len = int(patch_len)
        self.embed_dim = int(embed_dim)
        hidden = max(32, embed_dim // 2)
        self.time_branch = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, embed_dim),
        )
        self.freq_branch = nn.Sequential(
            nn.Linear(self.patch_len // 2 + 1, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, p = x.shape
        y = x.reshape(b * c * t, 1, p)
        time = self.time_branch(y)
        freq = torch.fft.rfft(x.float(), dim=-1).abs().reshape(b * c * t, -1)
        freq = self.freq_branch(freq).to(time.dtype)
        return self.drop(self.norm(time + freq)).view(b, c, t, self.embed_dim)
