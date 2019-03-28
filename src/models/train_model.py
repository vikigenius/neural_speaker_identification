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


def validate(hparams, dataset, model, progress, full=True):
    logger.info('Performing Valitation')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        shuffle=not full,
        num_workers=hparams.num_workers
    )
    correctness = []
    CeLoss = torch.nn.CrossEntropyLoss()
    batches = data_loader
    for batch in batches:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = torch_utils.to_var(v)
        if model.num_classes == 2:
            target = batch['gid']
        else:
            target = batch['cid']
        with torch.no_grad():
            logits = model(batch)
            pred = logits.argmax(1)
            loss = CeLoss(logits, target)
            correctness.append(pred == target)
        if not full:
            break
    acc = torch.cat(correctness).to(torch.float).mean()
    logger.info(f'Validation Acc = {acc}, Loss = {loss}')


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--resume', is_flag=True, help='Resume trainging from last ckpt')
@click.option('--progress', is_flag=True, help='Show progress bar')
@click.option('--gender', is_flag=True, help='Train Gender Classifier')
@click.pass_context
def train(ctx, dataset, resume, progress, gender):
    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams

    if gender:
        setattr(hparams, 'num_classes', 2)

    setattr(app_config, 'progress', progress)

    train_map_file = app_config.map_file[dataset].format('train')
    test_map_file = app_config.map_file[dataset].format('test')
    dataset = Spectrogram(train_map_file, 'train')

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=hparams.num_workers
    )

    val_dataset = Spectrogram(test_map_file, 'test')

    model = ResnetM(hparams)
    model.to(torch_utils.device)

    trainer = training_utils.Trainer(hparams, app_config, model)

    if resume:
        trainer.load_checkpoint()
    trainer.setup_optimizers(resume)

    validator = partial(validate, hparams, val_dataset, model, progress)

    for epoch in range(hparams.epochs):
        trainer.train(dataset, hparams.num_workers, data_loader,
                      validator=validator)
        trainer.save_checkpoint()
