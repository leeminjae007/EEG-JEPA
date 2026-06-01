import argparse
import random

import numpy as np
import torch
try:
    import yaml
except ImportError:
    yaml = None

from finetune import tuab_dataset
from finetune import tuab_model
from finetune import tuev_dataset
from finetune import tuev_model
from finetune import tusz_dataset
from finetune import tusz_model


DATASET_MODEL_REGISTRY = {
    'TUAB': (tuab_dataset, tuab_model),
    'TUEV': (tuev_dataset, tuev_model),
    'TUSZ': (tusz_dataset, tusz_model),
}


def parse_channel_names(text):
    if text is None or text == '':
        return None
    if isinstance(text, list):
        return [str(v).strip() for v in text if str(v).strip()]
    return [v.strip() for v in text.split(',') if v.strip()]


def parse_bool(text):
    return str(text).lower() in {'1', 'true', 'yes', 'y', 't'}


def parse_simple_yaml_mapping(text):
    def parse_scalar(raw):
        v = raw.strip()
        if v in {'true', 'True'}:
            return True
        if v in {'false', 'False'}:
            return False
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        if v.startswith('[') and v.endswith(']'):
            inner = v[1:-1].strip()
            return [] if inner == '' else [parse_scalar(item.strip()) for item in inner.split(',')]
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return v

    root = {}
    stack = [(-1, root)]

    for raw_line in text.splitlines():
        line_no_comment = raw_line.split('#', 1)[0].rstrip('\n').rstrip()
        if line_no_comment.strip() == '':
            continue

        indent = len(line_no_comment) - len(line_no_comment.lstrip(' '))
        line = line_no_comment.strip()
        if ':' not in line:
            continue

        while len(stack) > 0 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()

        if value == '':
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = parse_scalar(value)

    return root


def load_yaml_defaults(path, downstream_dataset, model_name):
    if path is None:
        return {}
    with open(path, 'r') as f:
        text = f.read()

    if yaml is not None:
        cfg = yaml.safe_load(text)
        if cfg is None:
            return {}
    else:
        cfg = parse_simple_yaml_mapping(text)

    if isinstance(cfg, dict) and ('common' in cfg or 'datasets' in cfg or 'models' in cfg):
        merged = {}
        common_cfg = cfg.get('common', {})
        if isinstance(common_cfg, dict):
            merged.update(common_cfg)

        dataset_cfg = cfg.get('datasets', {}).get(downstream_dataset, {})
        if isinstance(dataset_cfg, dict):
            merged.update(dataset_cfg)

        model_cfg = cfg.get('models', {}).get(model_name, {})
        if isinstance(model_cfg, dict):
            merged.update(model_cfg)
        return merged

    if isinstance(cfg, dict) and ('model' in cfg or 'hibrainmj' in cfg):
        model_cfg = cfg.get('model', {}) or {}
        merged = {
            'num_channels': model_cfg.get('n_channels', 19),
            'patch_size': model_cfg.get('patch_len', 16),
            'embed_dim': model_cfg.get('embed_dim', 768),
            'num_heads': model_cfg.get('num_heads', 12),
            'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
            'patch_criss_cross_depth': model_cfg.get('patch_criss_cross_depth', 4),
            'summary_depth': model_cfg.get('summary_depth', 1),
            'dropout': model_cfg.get('dropout', 0.1),
            'drop_path': model_cfg.get('drop_path', 0.0),
            'use_gated_summary_injection': model_cfg.get('use_gated_summary_injection', False),
            'montage_name': model_cfg.get('montage_name', 'standard_1020'),
            'channel_names': model_cfg.get('channel_names', None),
        }
        if model_cfg.get('spatial_pe_dim') is not None:
            merged['spatial_pe_dim'] = model_cfg['spatial_pe_dim']
        if model_cfg.get('temporal_pe_dim') is not None:
            merged['temporal_pe_dim'] = model_cfg['temporal_pe_dim']
        return merged

    return dict(cfg) if isinstance(cfg, dict) else {}


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--fname', type=str, default='configs/hibrainmj.yaml')
    pre_parser.add_argument('--downstream_dataset', type=str, default='TUAB')
    pre_parser.add_argument('--model_name', type=str, default='hibrainmj')
    pre_args, _ = pre_parser.parse_known_args()
    yaml_defaults = load_yaml_defaults(
        pre_args.fname,
        downstream_dataset=pre_args.downstream_dataset,
        model_name=pre_args.model_name,
    )

    parser = argparse.ArgumentParser(description='HiBrainMJ downstream finetune')
    parser.add_argument('--fname', type=str, default=pre_args.fname)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'SGD'])
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument(
        '--classifier',
        type=str,
        default='all_patch_reps',
        choices=['all_patch_reps', 'all_patch_reps_twolayer', 'all_patch_reps_onelayer', 'avgpooling_patch_reps'],
    )

    parser.add_argument('--downstream_dataset', type=str, default='TUAB')
    parser.add_argument('--datasets_dir', type=str, default='/data/datasets/TUAB/processed')
    parser.add_argument('--task_type', type=str, default='binaryclass', choices=['binaryclass', 'multiclass', 'regression'])
    parser.add_argument('--num_of_classes', type=int, default=2)
    parser.add_argument('--model_dir', type=str, default='/tmp/hibrainmj_finetune_tuab')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--multi_lr', type=parse_bool, default=True)
    parser.add_argument('--frozen', type=parse_bool, default=False)
    parser.add_argument('--use_pretrained_weights', type=parse_bool, default=True)
    parser.add_argument('--foundation_dir', type=str, default='')

    parser.add_argument('--model_name', type=str, default='hibrainmj')
    parser.add_argument('--num_channels', type=int, default=16)
    parser.add_argument('--num_patches', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=200)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--patch_criss_cross_depth', type=int, default=4)
    parser.add_argument('--summary_depth', type=int, default=1)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--use_gated_summary_injection', type=parse_bool, default=False)
    parser.add_argument('--spatial_pe_dim', type=int, default=None)
    parser.add_argument('--temporal_pe_dim', type=int, default=None)
    parser.add_argument('--montage_name', type=str, default='standard_1020')
    parser.add_argument(
        '--channel_names',
        type=str,
        default='',
        help='comma-separated channel names; empty means auto naming',
    )

    parser.set_defaults(**yaml_defaults)
    params = parser.parse_args()
    params.channel_names = parse_channel_names(params.channel_names)

    setup_seed(params.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(params.cuda)
        params.device = f'cuda:{params.cuda}'
    else:
        params.device = 'cpu'

    print(params)
    print('The downstream dataset is {}'.format(params.downstream_dataset))

    if params.downstream_dataset not in DATASET_MODEL_REGISTRY:
        print('Unsupported downstream dataset:', params.downstream_dataset)
        print('Available:', sorted(DATASET_MODEL_REGISTRY.keys()))
        return

    dataset_module, model_module = DATASET_MODEL_REGISTRY[params.downstream_dataset]
    load_dataset = dataset_module.LoadDataset(params)
    data_loader = load_dataset.get_data_loader()
    model = model_module.Model(params)
    from finetune_trainer import Trainer
    trainer = Trainer(params, data_loader, model)

    if params.task_type == 'multiclass':
        trainer.train_for_multiclass()
    elif params.task_type == 'regression':
        trainer.train_for_regression()
    else:
        trainer.train_for_binaryclass()

    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
