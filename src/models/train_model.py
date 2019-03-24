#!/usr/bin/env python
import click
from functools import partial
from torchvision.models import vgg11
from torch.utils.data import DataLoader
from src.data.dataset import Spectrogram
from src.utils import training_utils
from src.utils import torch_utils


@click.command()
@click.option('--resume', default=False, type=click.BOOL)
@click.option('--pbar', default=True, type=click.BOOL)
@click.pass_context
def train(ctx, resume, pbar):
    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams

    setattr(app_config, 'pbar', pbar)

    dataset = Spectrogram(app_config.feature_path,
                          app_config.qa_path, 'train')
    setattr(hparams, 'data_size', len(dataset))
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers
    )

    val_dataset = Spectrogram(app_config.feature_path,
                              app_config.qa_path, 'dev')

    model = vgg11(num_classes=1000)
    model.to(torch_utils.device)

    trainer = training_utils.Trainer(hparams, app_config, model)

    if resume:
        trainer.load_checkpoint()
    trainer.setup_optimizers(resume)

    validator = partial(validate_multric, hparams, val_dataset, model)

    for epoch in range(hparams.epochs):
        trainer.train(dataset, hparams.num_workers, data_loader, validator)
        trainer.save_checkpoint()
