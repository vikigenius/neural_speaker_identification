#!/usr/bin/env python
import os
import pickle
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class SInfo:
    cid: int = 0
    gid: int = 0
    path: str = ''


class M4AStreamer(object):
    def __init__(self, data_dir, extensions=['.wav', '.m4a']):
        self.extensions = extensions
        self.data_dir = data_dir

    def __iter__(self):
        for (dirpath, dirnames, files) in os.walk(self.data_dir,
                                                  followlinks=True):
            for filename in files:
                if any([filename.endswith(ext) for ext in self.extensions]):
                    yield os.path.join(dirpath, filename)

    def __len__(self):
        total_len = 0
        for (dirpath, dirnames, files) in os.walk(self.data_dir,
                                                  followlinks=True):
            for filename in files:
                if any([filename.endswith(ext) for ext in self.extensions]):
                    total_len += 1
        return total_len


class Spectrogram(Dataset):
    def __init__(self, map_file: str, split: str):
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
