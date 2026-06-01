# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import os
import pprint
from datetime import datetime

from src.utils.distributed import init_distributed
from src.train import main as app_main

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml(path):
    if yaml is None:
        raise RuntimeError('PyYAML is required to load config files')
    with open(path, 'r') as y_file:
        return yaml.load(y_file, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs/hibrainmj.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, fname, world_size, devices):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    import sys
    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = load_yaml(fname)
    logger.info('loaded params...')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(params)

    run_folder = os.environ.get('HIBRAINMJ_RUN_FOLDER')
    if run_folder:
        params.setdefault('logging', {})
        params['logging']['folder'] = run_folder
    resume_checkpoint = os.environ.get('HIBRAINMJ_RESUME_CHECKPOINT')
    if resume_checkpoint:
        params.setdefault('meta', {})
        params['meta']['load_checkpoint'] = True
        params['meta']['read_checkpoint'] = resume_checkpoint

    if world_size > 1:
        requested_world_size = world_size
        world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
        if world_size != requested_world_size:
            raise RuntimeError(
                'Distributed init failed for multi-GPU run. '
                f'Requested world_size={requested_world_size}, got world_size={world_size}. '
                'Check NCCL/CUDA build and launch with a single device until resolved.'
            )
        logger.info(f'Running... (rank: {rank}/{world_size})')
    else:
        logger.info('Running... (single-process)')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()
    normalized_devices = []
    for d in args.devices:
        d = str(d).strip()
        if d.startswith('cuda:'):
            normalized_devices.append(d)
        elif d.isdigit():
            normalized_devices.append(f'cuda:{d}')
        else:
            raise ValueError(
                f'Invalid device spec "{d}". Use values like cuda:0 cuda:1 ...'
            )
    args.devices = normalized_devices
    params = load_yaml(args.fname)
    base_folder = params.get('logging', {}).get('folder')
    if base_folder is not None:
        run_folder = os.environ.get('HIBRAINMJ_RUN_FOLDER')
        if run_folder is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_folder = os.path.join(base_folder, ts)
        os.makedirs(run_folder, exist_ok=True)
        os.environ['HIBRAINMJ_RUN_FOLDER'] = run_folder
        print(f'run-folder: {run_folder}')

    num_gpus = len(args.devices)
    if num_gpus == 1:
        process_main(0, args.fname, 1, args.devices)
    else:
        mp.set_start_method('spawn')
        procs = []
        for rank in range(num_gpus):
            p = mp.Process(
                target=process_main,
                args=(rank, args.fname, num_gpus, args.devices)
            )
            p.start()
            procs.append(p)

        exit_code = 0
        for p in procs:
            p.join()
            if p.exitcode not in (0, None):
                exit_code = p.exitcode

        if exit_code != 0:
            raise SystemExit(exit_code)
