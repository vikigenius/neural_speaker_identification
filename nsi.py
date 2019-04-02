#!/usr/bin/env python

import yaml
import logging.config
import click
import collections
import random
import numpy as np
import torch
from src.utils.params import Params
from src.features.build_features import featuregen
from src.models.train_model import train
from src.data.make_dataset import split


@click.group()
@click.option('--config', type=click.Path(exists=True), default='settings.yml')
@click.pass_context
def main(ctx, config):
    ctx.obj = collections.namedtuple('Config', ['app_config', 'hparams'])
    app_config = Params(config, 'defaults')
    hparams = Params(config, 'hparams')
    ctx.obj.app_config = app_config
    ctx.obj.hparams = hparams
    with open('logging_config.yml') as fp:
        log_cfg = yaml.safe_load(fp)
    logging.config.dictConfig(log_cfg)
    random.seed(1037)
    np.random.seed(99999)
    torch.manual_seed(1504)
    torch.cuda.manual_seed(1610)


main.add_command(featuregen)
main.add_command(train)
main.add_command(split)


if __name__ == '__main__':
    main()
