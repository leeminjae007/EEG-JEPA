import os
import torch
import numpy as np
from torch.utils.data import Dataset
from logging import getLogger
from tqdm import tqdm

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

logger = getLogger()


def _key_to_str(key):
    return key.decode('utf-8') if isinstance(key, bytes) else str(key)


def _subject_id_from_key(key):
    key = _key_to_str(key)
    if '_s' in key:
        return key.split('_s')[0]
    if '_' in key:
        return key.rsplit('_', 1)[0]
    return key


def _train_test_split(keys, test_ratio, seed):
    if train_test_split is not None:
        return train_test_split(keys, test_size=float(test_ratio), random_state=seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(keys))
    n_test = int(round(len(keys) * float(test_ratio)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return keys[train_idx], keys[test_idx]


class TUHEEGDataset(Dataset):
    def __init__(
        self,
        split='train',
        root_dir='/NFS71/EEG/tuh_eeg/tuh_eeg/tuh_eeg-preprocessed',
        n_channel=19,
        seq_length=6000,
        window_size=None,
        patch_size=None,
        test_ratio=0.2,
        seed=42,
        max_samples=None,
        num_subjects=None,
        subset_subjects_path=None,
        packed_data_path=None,
        packed_keys_path=None,
    ):
        """
        TUH EEG Dataset Loader
        
        Args:
            split: 'train', 'test' 중 선택
            root_dir: 전처리된 .npy 파일들이 저장된 경로
            n_channel: EEG 채널 수 (19)
            seq_length: 한 샘플의 길이 (6000 = 30초 * 200Hz)
            test_ratio: 테스트 데이터 비율
            downsample: 시간축 다운샘플링 여부
            sampling_rate: 다운샘플링 간격
            num_frames: 다운샘플링 후 프레임 수
        """
        self.root_dir = root_dir
        self.split = split
        self.n_channel = n_channel
        self.seq_length = seq_length
        self.window_size = window_size
        self.patch_size = patch_size
        self.packed_data_path = packed_data_path
        self.packed_keys_path = packed_keys_path
        self._packed_data = None
        
        # Load all keys
        keys_file = os.path.join(root_dir, 'all_keys.npy')
        if not os.path.exists(keys_file):
            raise FileNotFoundError(f"Keys file not found: {keys_file}")
        
        all_keys = np.load(keys_file, allow_pickle=True)
        logger.info(f'Loaded {len(all_keys)} keys from {keys_file}')

        if self.packed_data_path is None:
            candidate = os.path.join(root_dir, 'all_eeg.npy')
            if os.path.exists(candidate):
                self.packed_data_path = candidate
                self.packed_keys_path = os.path.join(root_dir, 'all_eeg_keys.npy')

        self.key_to_packed_index = None
        if self.packed_data_path is not None:
            if self.packed_keys_path is None:
                self.packed_keys_path = os.path.join(os.path.dirname(self.packed_data_path), 'all_eeg_keys.npy')
            packed_keys = np.load(self.packed_keys_path, allow_pickle=True)
            self.key_to_packed_index = {_key_to_str(k): i for i, k in enumerate(packed_keys)}
            logger.info(f'Using packed EEG memmap: {self.packed_data_path}')
        
        # Split data into train/test only (no val split in pretrain loader)
        if not (0.0 <= float(test_ratio) < 1.0):
            raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
        if float(test_ratio) == 0.0:
            train_keys = all_keys
            test_keys = all_keys[:0]
        else:
            train_keys, test_keys = _train_test_split(all_keys, test_ratio, seed)
        
        # Select split
        if split == 'train':
            self.keys = train_keys
        elif split == 'test':
            self.keys = test_keys
        else:
            raise ValueError(f"Invalid split: {split}. Choose from 'train' or 'test'")

        if subset_subjects_path is not None:
            if not os.path.exists(subset_subjects_path):
                raise FileNotFoundError(f"subset_subjects_path not found: {subset_subjects_path}")
            with open(subset_subjects_path, 'r') as f:
                fixed_subjects = [line.strip() for line in f if line.strip()]
            fixed_subjects = set(fixed_subjects)
            subject_ids = np.array([_subject_id_from_key(k) for k in self.keys], dtype=object)
            keep_mask = np.array([sid in fixed_subjects for sid in subject_ids])
            self.keys = self.keys[keep_mask]
            logger.info(f'Using fixed subject list from {subset_subjects_path} ({len(fixed_subjects)} subjects)')
        elif num_subjects is not None:
            num_subjects = int(num_subjects)
            if num_subjects <= 0:
                raise ValueError(f"num_subjects must be > 0, got {num_subjects}")

            subject_ids = np.array([_subject_id_from_key(k) for k in self.keys], dtype=object)
            unique_subjects = sorted(np.unique(subject_ids).tolist())
            selected_subjects = set(unique_subjects[:num_subjects])
            keep_mask = np.array([sid in selected_subjects for sid in subject_ids])
            self.keys = self.keys[keep_mask]
            logger.info(f'Using first {num_subjects} sorted subjects for {split} split')

        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError(f"max_samples must be > 0, got {max_samples}")
            if len(self.keys) > max_samples:
                rng = np.random.RandomState(seed)
                indices = rng.permutation(len(self.keys))[:max_samples]
                self.keys = self.keys[indices]
                logger.info(f'Using max_samples={max_samples} for {split} split')
        
        logger.info(f'Loaded {len(self.keys)} keys for {split} split')

    def __len__(self):
        return len(self.keys)

    def _load_packed(self):
        if self._packed_data is None:
            self._packed_data = np.load(self.packed_data_path, mmap_mode='r')
        return self._packed_data

    def __getitem__(self, idx):
            # 1. 파일 키 가져오기 및 경로 설정
            key = self.keys[idx]
            key_str = _key_to_str(key)

            if self.key_to_packed_index is not None:
                data = self._load_packed()[self.key_to_packed_index[key_str]]
            else:
                file_path = os.path.join(self.root_dir, f'{key_str}.npy')
                data = np.load(file_path, allow_pickle=False)
            data = np.array(data, dtype=np.float32, copy=True)
            
            if self.window_size is not None:
                if data.shape[1] < self.window_size:
                    raise ValueError(
                        f'window_size={self.window_size} is larger than sample length {data.shape[1]}')
                data = data[:, :self.window_size]

            if self.patch_size is not None:
                if data.shape[1] % self.patch_size != 0:
                    raise ValueError(
                        f'window_size ({data.shape[1]}) must be divisible by patch_size ({self.patch_size})')
                t = data.shape[1] // self.patch_size
                data = data.reshape(data.shape[0], t, self.patch_size)

            # 3. 넘파이를 토치 텐서로 변환
            data = torch.from_numpy(data)

            # 4. 모델이 기대하는 (Channel, T, P) 형태로 반환
            # 학습 코드의 udata['eeg'] 호출에 맞춰 키값을 'eeg'로 설정
            return {
                'eeg': data,
                'id': key_str
            }

    def _temporal_sampling(self, frames, start_idx, end_idx, num_samples):
        """Sample num_samples frames between start_idx and end_idx"""
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, frames.shape[2] - 1).long()
        new_frames = torch.index_select(frames, 2, index)
        return new_frames


def make_tuheeg(
    batch_size,
    split='train',
    root_dir='/NFS71/EEG/tuh_eeg/tuh_eeg/tuh_eeg-preprocessed',
    window_size=None,
    patch_size=None,
    test_ratio=0.2,
    collator=None,
    pin_mem=True,
    num_workers=4,
    persistent_workers=False,
    loader_timeout=0,
    world_size=1,
    rank=0,
    drop_last=True,
    seed=42,
    max_samples=None,
    num_subjects=None,
    subset_subjects_path=None,
    packed_data_path=None,
    packed_keys_path=None,
    prefetch_factor=None,
):
    """Create TUH EEG dataset and dataloader"""
    dataset = TUHEEGDataset(
        split=split,
        root_dir=root_dir,
        window_size=window_size,
        patch_size=patch_size,
        test_ratio=test_ratio,
        seed=seed,
        max_samples=max_samples,
        num_subjects=num_subjects,
        subset_subjects_path=subset_subjects_path,
        packed_data_path=packed_data_path,
        packed_keys_path=packed_keys_path,
    )
    logger.info(f'TUH EEG {split} dataset created with {len(dataset)} samples')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == 'train'))
    
    loader_kwargs = {
        'collate_fn': collator,
        'sampler': dist_sampler,
        'batch_size': batch_size,
        'drop_last': drop_last,
        'pin_memory': pin_mem,
        'num_workers': num_workers,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        loader_kwargs['timeout'] = loader_timeout
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = int(prefetch_factor)

    data_loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)
    
    logger.info(f'TUH EEG {split} data loader created')
    return dataset, data_loader, dist_sampler


if __name__ == '__main__':
    print('Loading TUH EEG dataset...')
    
    # Test dataset loading
    dataset = TUHEEGDataset(split='train')
    print(f'Train dataset size: {len(dataset)}')
    
    # Test single sample
    sample = dataset[0]
    print(f'Sample EEG shape: {sample["eeg"].shape}')  # Should be (1, 19, 6000)
    print('Dataset loaded successfully!')
    
    # Test dataloader
    train_dataset, train_loader, train_sampler = make_tuheeg(
        batch_size=4,
        split='train',
        num_workers=0
    )
    for batch in train_loader:
        print(f'Batch EEG shape: {batch["eeg"].shape}')  # Should be (4, 1, 19, 6000)
        break
