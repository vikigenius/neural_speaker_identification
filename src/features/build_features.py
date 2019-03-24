#!/usr/bin/env python

import click
import logging
import os
import pickle
from tqdm import tqdm
import numpy as np
from src.features.spectrum import Spectrum
from src.data.dataset import M4AStreamer


logger = logging.getLogger(__name__)


@click.command()
@click.option('--split', default='train', type=click.Choice(['train', 'test']))
@click.option('--duration', default=3.0,
              help='Duration of Audio files to extract')
@click.option('--verbose', '-v', is_flag=True, help='show debug output')
@click.option('--progress', is_flag=True, help='Show Progress Bar')
@click.pass_context
def featuregen(ctx, split, duration, verbose, progress):
    if verbose:
        logger.setLevel(logging.DEBUG)

    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams
    setattr(hparams, 'duration', duration)

    data_dir = app_config.data_dir.format(split)
    audio_files = M4AStreamer(data_dir)

    specgen = Spectrum(hparams)

    dset_list = []

    if progress and not verbose:
        audio_files = tqdm(audio_files)
    fcount = 0
    for cid, audio_file in audio_files:
        fcount += 1
        spec = specgen.generate(audio_file)
        fname = os.path.splitext(audio_file)[0] + '.npy'
        np.save(fname, spec)
        dset_list.append((cid, fname))
        logger.debug(
            f'{spec.shape} Spectrogram created in {fname} for id {cid}')
        if not progress and (fcount + 1) % 100 == 0:
            logger.info(f'{fcount} spectrograms created')

    mapfile_name = app_config.map_file.format(split)
    with open(mapfile_name, 'wb') as f:
        pickle.dump(dset_list, f)
    logger.info(f'Mapping file created at {mapfile_name}')
