#!/usr/bin/env python
import click
import torch
import logging
from functools import partial
from torch.utils.data import DataLoader
from src.data.dataset import Spectrogram
from src.utils import training_utils
from src.utils import torch_utils
from src.models.resnet import ResnetM


logger = logging.getLogger(__name__)


def validate(hparams, dataset, model):
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers
    )
    correctness = []
    softmax = torch.nn.Softmax()
    for batch in data_loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = torch_utils.to_var(v)

        with torch.no_grad():
            pred = model(batch)
            pred = softmax(pred).argmax(1)
            correctness.append(pred == batch['cid'])
    acc = torch.stack(correctness).mean()
    logger.info(f'Validation Acc = {acc}')


@click.command()
@click.option('--resume', default=False, type=click.BOOL)
@click.option('--pbar', default=True, type=click.BOOL)
@click.pass_context
def train(ctx, resume, pbar):
    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams

    setattr(app_config, 'pbar', pbar)

    train_map_file = app_config.map_file.format('train')
    dataset = Spectrogram(train_map_file, 'train')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers
    )

    val_dataset = Spectrogram(train_map_file, 'test')

    model = ResnetM(hparams)
    model.to(torch_utils.device)

    trainer = training_utils.Trainer(hparams, app_config, model)

    if resume:
        trainer.load_checkpoint()
    trainer.setup_optimizers(resume)

    validator = partial(validate, hparams, val_dataset, model)

    for epoch in range(hparams.epochs):
        trainer.train(dataset, hparams.num_workers, data_loader,
                      validator=validator)
        trainer.save_checkpoint()
