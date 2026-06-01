import os
import random

import mne
import numpy as np
from tqdm import tqdm


selected_channels = {
    '01_tcp_ar': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ],
    '02_tcp_le': [
            'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
            'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
            'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
    ],
    '03_tcp_ar_a': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ]
}

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def iter_files(rootDir):
    file_path_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            # print(file_name)
            file_path_list.append(file_name)
    return file_path_list

def preprocessing_recording(file_path, file_key_list: list, save_dir: str):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    if '02_tcp_le' in file_path:
        channel_key = '02_tcp_le'
    elif '01_tcp_ar' in file_path:
        channel_key = '01_tcp_ar'
    elif '03_tcp_ar_a' in file_path:
        channel_key = '03_tcp_ar_a'
    else:
        return
    channel_list = selected_channels[channel_key]
    for ch in channel_list:
        if ch not in raw.info['ch_names']:
            return
    raw.pick_channels(channel_list, ordered=True)
    # print(raw.info)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))
    eeg_array = raw.to_data_frame().values
    # print(raw.info)
    eeg_array = eeg_array[:, 1:]
    points, chs = eeg_array.shape
    if points < 300 * 200:
        return
    a = points % (30 * 200)
    eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
    # print(eeg_array.shape)
    eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    eeg_array = eeg_array.transpose(0, 3, 1, 2)
    print(eeg_array.shape)
    file_name = file_path.split('/')[-1][:-4]

    for i, sample in enumerate(eeg_array):
        # print(i, sample.shape)
        if np.max(np.abs(sample)) < 100:
            sample_key = f'{file_name}_{i}'
            print(sample_key)
            file_key_list.append(sample_key)
            # Reshape (channels, 30, 200) -> (channels, 6000)
            sample_reshaped = sample.reshape(sample.shape[0], -1)
            np.save(os.path.join(save_dir, f"{sample_key}.npy"), sample_reshaped)

if __name__ == '__main__':
    setup_seed(42)
    file_path_list = iter_files(r"/NFS/EEG/tuh_eeg/tuh_eeg/v2.0.1/edf/")
    # .edf 파일만 필터링
    file_path_list = [f for f in file_path_list if f.endswith('.edf')]
    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    save_dir = r"/NFS/EEG/tuh_eeg/tuh_eeg/tuh_eeg-preprocessed"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(file_path_list)} .edf files")
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, save_dir)
    # 전체 key 리스트 저장
    np.save(os.path.join(save_dir, "all_keys.npy"), np.array(file_key_list))
    print(f"Saved {len(file_key_list)} samples")
