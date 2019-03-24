#!/usr/bin/env python

import yaml
import logging.config
import click
import collections
from src.utils.params import Params
from src.features.build_features import featuregen


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
        log_cfg = yaml.load(fp)
    logging.config.dictConfig(log_cfg)


main.add_command(featuregen)


if __name__ == '__main__':
    main()
