#!/usr/bin/env python
import pickle
import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from src.features.raw import ProcessedRaw


@dataclass
class SInfo:
    cid: int = 0
    gid: int = 0
    path: str = ''


class CelebSpeech(Dataset):
    def __init__(self, map_file: str, tdur=None):
        """
        Spectrogram Dataset
        Args:
            map_file: Path of raw audio
            tdur: int, the duration to truncate it to
        """
        with open(map_file, 'rb') as f:
            self.spec_list = pickle.load(f)
        self.tdur = tdur
        self.processor = ProcessedRaw(16000.0, preprocess=True)

    def __getitem__(self, idx):
        sinfo = self.spec_list[idx]
        path = sinfo.path
        if self.tdur:
            raw = self.processor.load_sample(path, self.tdur)
        else:
            raw = self.processor.load(path)
        return {
            'cid': sinfo.cid,
            'gid': sinfo.gid,
            'raw': raw,
        }

    def __len__(self):
        return len(self.spec_list)


class RawSpeech(Dataset):
    def __init__(self, map_file: str, duration: int):
        with open(map_file, 'rb') as f:
            self.spec_list = pickle.load(f)

        self.duration = duration
        self.raw_audio = ProcessedRaw(16000.0, duration)

    def __getitem__(self, idx):
        sinfo = self.spec_list[idx]
        raw_sample = self.raw_audio.load(sinfo.path)
        af = np.random.uniform(0.8, 1.2)
        raw_sample *= af  # Random Amp noise to help avoid overfitting
        return {
            'cid': sinfo.cid,
            'gid': sinfo.gid,
            'raw': raw_sample
        }

    def __len__(self):
        return len(self.spec_list)


class RawSpeechChunks(object):
    def __init__(self, map_file: str, duration, overlap, batch_size,
                 shuffle=True):
        with open(map_file, 'rb') as f:
            self.spec_list = pickle.load(f)

        self.index = np.arange(len(self))
        if shuffle:
            np.random.shuffle(self.index)

        self.batch_size = batch_size
        self.raw_audio = ProcessedRaw(16000.0, duration, overlap, batch_size)

    def __getitem__(self, idx):
        ridx = self.index[idx]
        sinfo = self.spec_list[ridx]
        raw_chunks = self.raw_audio.load(sinfo.path)
        raw_chunks = torch.from_numpy(raw_chunks)
        batch_size = raw_chunks.size(0)
        cids = torch.zeros(batch_size) + sinfo.cid
        gids = torch.zeros(batch_size) + sinfo.gid

        return {
            'cid': cids.long(),
            'gid': gids.long(),
            'raw': raw_chunks.float()
        }

    def __len__(self):
        return len(self.spec_list)
