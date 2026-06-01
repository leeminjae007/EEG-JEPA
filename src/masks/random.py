from dataclasses import dataclass
from multiprocessing import Value
from typing import Optional, Sequence, Tuple

import torch


@dataclass
class EEGMaskBatch:
    """Rectangular HiBrainMJ masks on an EEG [C, T] patch grid."""

    context_mask: torch.Tensor
    target_mask: torch.Tensor
    context_indices: torch.Tensor
    target_indices: torch.Tensor

    def to(self, device):
        return EEGMaskBatch(
            context_mask=self.context_mask.to(device),
            target_mask=self.target_mask.to(device),
            context_indices=self.context_indices.to(device),
            target_indices=self.target_indices.to(device),
        )


class BlockMaskGenerator:
    """HiBrainMJ-style random rectangular blocks over an EEG channel x time grid."""

    def __init__(
        self,
        input_size: Optional[Tuple[int, int]] = None,
        target_scale: Sequence[float] = (0.15, 0.2),
        context_scale: Sequence[float] = (0.85, 1.0),
        aspect_ratio: Sequence[float] = (0.75, 1.5),
        num_target_blocks: int = 4,
        num_context_blocks: int = 1,
        min_keep: int = 1,
        seed: int = 42,
    ):
        self.input_size = tuple(input_size) if input_size is not None else None
        self.target_scale = tuple(float(v) for v in target_scale)
        self.context_scale = tuple(float(v) for v in context_scale)
        self.aspect_ratio = tuple(float(v) for v in aspect_ratio)
        self.num_target_blocks = int(num_target_blocks)
        self.num_context_blocks = int(num_context_blocks)
        self.min_keep = int(min_keep)
        self.seed = int(seed)
        self._itr_counter = Value("i", -1)

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return int(self._itr_counter.value)

    @staticmethod
    def _uniform(bounds, generator):
        lo, hi = bounds
        return lo + torch.rand((), generator=generator).item() * (hi - lo)

    def _block_size(self, c, t, scale, generator):
        area = max(1, int(round(c * t * self._uniform(scale, generator))))
        ar = self._uniform(self.aspect_ratio, generator)
        h = max(1, min(c, int(round((area * ar) ** 0.5))))
        w = max(1, min(t, int(round((area / ar) ** 0.5))))
        return h, w

    @staticmethod
    def _rect(c, t, h, w, generator):
        top = torch.randint(0, c - h + 1, (), generator=generator).item()
        left = torch.randint(0, t - w + 1, (), generator=generator).item()
        mask = torch.zeros(c, t, dtype=torch.bool)
        mask[top:top + h, left:left + w] = True
        return mask

    def _blocks(self, c, t, count, scale, generator, allowed=None):
        out = torch.zeros(c, t, dtype=torch.bool)
        allowed = torch.ones(c, t, dtype=torch.bool) if allowed is None else allowed.bool()
        for _ in range(int(count)):
            best, best_count = None, -1
            h, w = self._block_size(c, t, scale, generator)
            for _ in range(20):
                block = self._rect(c, t, h, w, generator) & allowed
                n = int(block.sum().item())
                if n > best_count:
                    best, best_count = block, n
                if n >= self.min_keep:
                    break
            out |= best
        return out & allowed

    @staticmethod
    def _indices(mask):
        return torch.nonzero(mask.flatten(), as_tuple=False).flatten().long()

    @classmethod
    def _stack_indices(cls, masks):
        idx = [cls._indices(m) for m in masks]
        width = max(1, max((int(i.numel()) for i in idx), default=0))
        out = torch.full((len(idx), width), -1, dtype=torch.long)
        for i, values in enumerate(idx):
            out[i, :values.numel()] = values
        return out

    def __call__(self, batch_size, input_size=None, device=None):
        c, t = tuple(input_size or self.input_size)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.step())
        contexts, targets = [], []
        for _ in range(int(batch_size)):
            target = self._blocks(c, t, self.num_target_blocks, self.target_scale, generator)
            context = self._blocks(c, t, self.num_context_blocks, self.context_scale, generator, allowed=~target)
            contexts.append(context if int(context.sum().item()) >= self.min_keep else ~target)
            targets.append(target)
        batch = EEGMaskBatch(
            context_mask=torch.stack(contexts),
            target_mask=torch.stack(targets),
            context_indices=self._stack_indices(contexts),
            target_indices=self._stack_indices(targets),
        )
        return batch.to(device) if device is not None else batch


class EEGMaskCollator:
    """DataLoader collator for the current HiBrainMJ random block rule."""

    def __init__(self, input_size=None, patch_len=16, **kwargs):
        self.input_size = tuple(input_size) if input_size is not None else None
        self.patch_len = int(patch_len)
        self.generator = BlockMaskGenerator(input_size=input_size, **kwargs)

    def _infer_grid(self, eeg):
        if self.input_size is not None:
            return self.input_size
        if eeg.ndim == 4:
            return int(eeg.shape[1]), int(eeg.shape[2])
        return int(eeg.shape[1]), int(eeg.shape[2]) // self.patch_len

    @staticmethod
    def _crop(indices):
        width = int(indices.ge(0).sum(dim=1).min().item())
        return torch.stack([row[row.ge(0)][:width] for row in indices], dim=0)

    def build_mask_batch_from_eeg(self, eeg):
        return self.generator(eeg.shape[0], input_size=self._infer_grid(eeg), device=eeg.device)

    def build_masks_from_eeg(self, eeg):
        masks = self.build_mask_batch_from_eeg(eeg)
        return [self._crop(masks.context_indices)], [self._crop(masks.target_indices)]

    def __call__(self, batch):
        collated = torch.utils.data.default_collate(batch)
        eeg = collated["eeg"] if isinstance(collated, dict) else collated
        masks_enc, masks_pred = self.build_masks_from_eeg(eeg)
        return collated, masks_enc, masks_pred


def build_mask_collator(mask_cfg, input_size):
    return EEGMaskCollator(
        input_size=input_size,
        patch_len=int(mask_cfg.get("patch_len", mask_cfg.get("patch_size", 16))),
        target_scale=mask_cfg.get("target_scale", (0.15, 0.2)),
        context_scale=mask_cfg.get("context_scale", (0.85, 1.0)),
        aspect_ratio=mask_cfg.get("aspect_ratio", (0.75, 1.5)),
        num_target_blocks=int(mask_cfg.get("num_target_blocks", 4)),
        num_context_blocks=int(mask_cfg.get("num_context_blocks", 1)),
        min_keep=int(mask_cfg.get("min_keep", 1)),
        seed=int(mask_cfg.get("seed", 42)),
    )


def apply_masks(x, masks):
    out = []
    for mask in masks:
        index = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        out.append(torch.gather(x, dim=1, index=index))
    return torch.cat(out, dim=0)


MaskCollator = EEGMaskCollator
