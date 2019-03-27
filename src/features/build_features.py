#!/usr/bin/env python

import click
import logging
import os
import pickle
from tqdm import tqdm
import numpy as np
from src.features.spectrum import Spectrum
from src.data.dataset import M4AStreamer, SInfo


logger = logging.getLogger(__name__)


def validate_gen(specgen, min_dur, audio_file):
    spec = specgen.generate(audio_file)
    if spec.shape[1] < min_dur:
        logger.warn(
            f'Shape Mismatch for spec1 = {spec.shape} in {audio_file}')
    return spec


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--split', default='train', type=click.Choice(['train', 'test']))
@click.option('--duration', default=3.0,
              help='Duration of Audio files to extract')
@click.option('--verbose', '-v', is_flag=True, help='show debug output')
@click.option('--progress', is_flag=True, help='Show Progress Bar')
@click.option('--rebuild', is_flag=True, help='Rebuild File List')
@click.option('--force', is_flag=True, help='Force overwrite spectrograms')
@click.pass_context
def featuregen(ctx, dataset, split, duration, verbose, progress, rebuild,
               force):
    if verbose:
        logger.setLevel(logging.DEBUG)

    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams
    setattr(hparams, 'duration', duration)

    data_dir = app_config.data_dir.format(dataset, split)
    audio_files = M4AStreamer(data_dir, dataset)

    specgen = Spectrum(hparams)

    dset_list = []
    mapfile_name = app_config.map_file.format(split)
    if not force and os.path.exists(mapfile_name):
        with open(mapfile_name, 'rb') as f:
            dset_list = pickle.load(f)
    rebuild_list = True if dset_list == [] else False
    rebuild = rebuild_list or rebuild
    if progress and not verbose:
        audio_files = tqdm(audio_files)
    fcount = 0

    min_dur = 301

    for cid, gid, audio_file in audio_files:
        fname = os.path.splitext(audio_file)[0] + '.npy'
        if not force and os.path.isfile(fname):
            if rebuild:
                dset_list.append(fname)
            continue

        fcount += 1

        spec1 = validate_gen(specgen, min_dur, audio_file)

        np.save(fname, spec1)
        dset_list.append(SInfo(cid, gid, fname))

        logger.debug(
            f'{spec1.shape} Spectrogram created in {fname} for id {cid}')
        if not progress and (fcount + 1) % 50000 == 0:
            logger.info(f'{fcount} spectrograms created')
    mapfile_name = app_config.map_file.format(split)
    with open(mapfile_name, 'wb') as f:
        pickle.dump(dset_list, f)
    logger.info(f'Mapping file created at {mapfile_name}')
