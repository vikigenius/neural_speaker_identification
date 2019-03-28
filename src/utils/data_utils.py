#!/usr/bin/env python
import os


def get_hash(path: str):
    hashpath = os.path.dirname(path)
    return os.path.basename(hashpath)


def get_cid(path: str):
    idpath = os.path.dirname(os.path.dirname(path))
    cidstr = os.path.basename(idpath)
    cid = int(cidstr.replace('id1', '').replace('id0', ''))
    return cid - 1


def get_pid(path: str):
    idpath = os.path.dirname(os.path.dirname(path))
    cidstr = os.path.basename(idpath)
    return cidstr


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
