#!/usr/bin/env python

import click
import logging
import os
from tqdm import tqdm
import numpy as np
from src.features.spectrum import Spectrum
from src.utils.data_utils import M4AStreamer


logger = logging.getLogger(__name__)


def validate_and_save(sgrams, dur, base_name: str):
    for i, sgram in enumerate(sgrams):
        assert sgram.shape[1] == dur
        fname = base_name + '_' + str(i) + '.npy'
        np.save(fname, sgram)


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--duration', default=3.0,
              help='Duration of Audio files to extract')
@click.option('--verbose', '-v', is_flag=True, help='show debug output')
@click.option('--progress', is_flag=True, help='Show Progress Bar')
@click.option('--force', is_flag=True, help='Force overwrite spectrograms')
@click.pass_context
def featuregen(ctx, dataset, duration, verbose, progress, force):
    if verbose:
        logger.setLevel(logging.DEBUG)

    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams
    setattr(hparams, 'duration', duration)

    data_dir = app_config.data_dir[dataset]
    audio_files = M4AStreamer(data_dir)

    specgen = Spectrum(hparams)

    if progress and not verbose:
        audio_files = tqdm(audio_files)

    for audio_file in audio_files:
        base_name = os.path.splitext(audio_file)[0]

        sgrams = specgen.generate(audio_file)

        validate_and_save(sgrams, 301, base_name)

    logger.info('Finished generating all spectrograms')
