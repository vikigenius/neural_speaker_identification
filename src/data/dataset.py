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


class Spectrogram(Dataset):
    def __init__(self, map_file: str):
        with open(map_file, 'rb') as f:
            self.spec_list = pickle.load(f)

    def __getitem__(self, idx):
        sinfo = self.spec_list[idx]
        sgram = np.load(sinfo.path)
        sgram -= np.mean(sgram, 1, keepdims=True)
        sgram /= np.std(sgram, 1, keepdims=True)
        return {
            'cid': sinfo.cid,
            'gid': sinfo.gid,
            'sgram': sgram
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
            np.shuffle(self.index)

        self.batch_size = batch_size
        self.raw_audio = ProcessedRaw(16000.0, duration, overlap, batch_size)

    def __getitem__(self, idx):
        ridx = self.index[idx]
        sinfo = self.spec_list[ridx]
        raw_chunks = self.raw_audio.load(sinfo.path)
        raw_chunks = torch.from_numpy(raw_chunks)
        cids = torch.zeros_like(raw_chunks) + sinfo.cid
        gids = torch.zeros_like(raw_chunks) + sinfo.gid

        return {
            'cid': cids,
            'gid': gids,
            'raw': raw_chunks
        }

    def __len__(self):
        return len(self.spec_list)
