#!/usr/bin/env python
import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class M4AStreamer(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __iter__(self):
        for (dirpath, dirnames, files) in os.walk(self.root_dir,
                                                  followlinks=True):
            for filename in files:
                if filename.endswith('.m4a'):
                    pdir = os.path.dirname(dirpath)
                    pdir = os.path.basename(pdir)
                    cid = int(pdir.replace('id', ''))
                    yield cid, os.path.join(dirpath, filename)

    def __len__(self):
        total_len = 0
        for (dirpath, dirnames, files) in os.walk(self.root_dir,
                                                  followlinks=True):
            for filename in files:
                if filename.endswith('.m4a'):
                    total_len += 1
        return total_len


class Spectrogram(Dataset):
    def __init__(self, map_file: str, split: str):
        with open(map_file, 'rb') as f:
            self.spec_list = pickle.load(f)

    def __getitem__(self, idx):
        cid, filename = self.spec_list[idx]
        sgram = np.load(filename)
        return {
            'cid': cid,
            'sgram': sgram
        }

    def __len__(self):
        return len(self.spec_list)
