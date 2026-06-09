# HiBrainMJ

HiBrainMJ is an EEG JEPA-style representation learning codebase. It keeps the
I-JEPA idea of predicting masked target latents from visible context, but uses an
EEG-specific patch encoder, positional encoding, and masking rule.

## Model

- Input EEG: `[B, C, L]`
- Patch grid: `[B, C, T, patch_len]`
- Default EEG setup: `C=19`, `L=6000`, `patch_len=200`, `T=30`
- Patch token shape: `[B, C, T, D]`

Current backbone settings follow the CBraMod scale where possible:

- `embed_dim=200`
- `num_heads=8`
- temporal attention heads `4`
- spatial attention heads `4`
- FFN hidden dim `800`
- criss-cross layers `12`

The patch encoder uses a CBraMod-style time-frequency encoder:

- time branch: 3-layer Conv2d over raw patches
- frequency branch: `torch.fft.rfft` magnitude + `Linear(101, 200)`
- output: `[B, C, T, 200]`

Positional encoding is HiBrainMJ-specific:

- 3D standard 10-20 montage coordinates for spatial PE
- sinusoidal temporal PE
- final PE: spatial + temporal, currently `100 + 100 = 200`

## Masking

The default pretraining mask strategy is `brain`.

Channel masking is based on physical 10-20 electrode geometry, not saved channel
order. Target channel blocks are sampled as 3D coordinate spheres on the scalp.
Time masking keeps the Brain-JEPA-style temporal rule.

The current default uses full context:

```text
context = full EEG patch grid - target blocks
target  = spatial 10-20 block + temporal block
```

This prevents target leakage while allowing the context encoder to see the full
visible EEG context.

## Pretraining

Config:

```bash
configs/hibrainmj.yaml
```

Run:

```bash
python main.py \
  --fname configs/hibrainmj.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3
```

For background execution:

```bash
mkdir -p /home/leemj/new_vis/logs/hibrainmj
nohup python main.py \
  --fname configs/hibrainmj.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 \
  > /home/leemj/new_vis/logs/hibrainmj/pretrain.out 2>&1 &

tail -f /home/leemj/new_vis/logs/hibrainmj/pretrain.out
```

Checkpoints are saved every `logging.save_interval` epochs and as
`hibrainmj-latest.pth.tar`.

## Finetuning

Config:

```bash
configs/finetune_tuab.yaml
```

Run:

```bash
python finetune_main.py --fname configs/finetune_tuab.yaml
```

TUAB supports packed memmap cache through:

```yaml
packed_cache_dir: /tmp/hibrainmj_finetune_cache
rebuild_packed_cache: false
```

If the cache exists, finetuning loads the memmap cache directly instead of
listing all original pickle files.

## Notes

- Old 768-dim checkpoints are not compatible with the current 200-dim backbone.
- After changing model dimensions, pretrain again before using pretrained
  finetune weights.
- W&B logging is configured in the YAML files.
