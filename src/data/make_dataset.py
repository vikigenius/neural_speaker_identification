# -*- coding: utf-8 -*-
import click
import logging
import pickle
import pandas as pd
from tqdm import tqdm
from src.utils import data_utils
from src.data.dataset import SInfo


logger = logging.getLogger(__name__)


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--raw', is_flag=True, help='Split the raw samples')
@click.option('--verbose', '-v', is_flag=True, help='show debug output')
@click.option('--progress', is_flag=True, help='Show Progress Bar')
@click.option('--force', is_flag=True, help='Force overwrite spectrograms')
@click.pass_context
def split(ctx, dataset, raw, verbose, progress, force):
    if verbose:
        logger.setLevel(logging.DEBUG)

    app_config = ctx.obj.app_config
    num_classes = app_config.num_classes

    data_dir = app_config.data_dir[dataset]

    if raw:
        extensions = ['.wav']
        file_type = 'raw'
    else:
        extensions = ['.npy']
        file_type = 'sgram'

    data_files = data_utils.M4AStreamer(data_dir, extensions=extensions)

    if progress and not verbose:
        data_files = tqdm(data_files)

    idmap = [None]*num_classes

    train_list = []
    test_list = []

    meta_file = app_config.meta_file[dataset]
    meta_info = pd.read_csv(meta_file, delim_whitespace=True)
    meta_info = meta_info.set_index('ID')

    for sgramfile in data_files:
        cid = data_utils.get_cid(sgramfile)
        thash = data_utils.get_hash(sgramfile)
        if idmap[cid] is None:
            idmap[cid] = thash

        # Compute Info
        pid = data_utils.get_pid(sgramfile)
        gender = meta_info['Gender'][pid]

        gender = gender == 'm'

        info = SInfo(cid, gender, sgramfile)

        if idmap[cid] == thash:
            test_list.append(info)
        else:
            train_list.append(info)

    map_file = app_config.map_file[dataset]
    train_file = map_file.format(file_type, 'train')
    test_file = map_file.format(file_type, 'test')

    with open(train_file, 'wb') as trainf:
        pickle.dump(train_list, trainf)

    with open(test_file, 'wb') as testf:
        pickle.dump(test_list, testf)

    logger.info(f'Training Map created at {train_file}')
    logger.info(f'Testing Map created at {test_file}')
