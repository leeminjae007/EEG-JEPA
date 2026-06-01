import argparse
import os

import numpy as np
from tqdm import tqdm


def _key_to_str(key):
    return key.decode("utf-8") if isinstance(key, bytes) else str(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--keys-file", default="all_keys.npy")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    keys = np.load(os.path.join(args.root_dir, args.keys_file), allow_pickle=True)
    first = np.load(os.path.join(args.root_dir, f"{_key_to_str(keys[0])}.npy"), allow_pickle=False)
    if args.window_size is not None:
        first = first[:, :args.window_size]

    data_path = os.path.join(args.out_dir, "all_eeg.npy")
    keys_path = os.path.join(args.out_dir, "all_eeg_keys.npy")
    data = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.dtype(args.dtype),
        shape=(len(keys),) + tuple(first.shape),
    )

    for i, key in enumerate(tqdm(keys, desc="packing TUH EEG")):
        arr = np.load(os.path.join(args.root_dir, f"{_key_to_str(key)}.npy"), allow_pickle=False)
        if args.window_size is not None:
            arr = arr[:, :args.window_size]
        data[i] = arr.astype(args.dtype, copy=False)

    data.flush()
    np.save(keys_path, keys)
    print(f"packed_data_path: {data_path}")
    print(f"packed_keys_path: {keys_path}")


if __name__ == "__main__":
    main()

