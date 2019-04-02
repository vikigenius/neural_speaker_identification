#!/usr/bin/env python
import click
import torch
import logging
from functools import partial
from torch.utils.data import DataLoader
from src.data.dataset import CelebSpeech, RawSpeech, RawSpeechChunks
from src.utils import training_utils
from src.utils import torch_utils
from src.models.specnet import SpecNet
from src.models.speechnet import SpeechNet


logger = logging.getLogger(__name__)


def validate(hparams, dataset, model, progress, full=True):
    logger.info('Performing Valitation')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=not full,
        pin_memory=True,
        num_workers=hparams.num_workers
    )
    correctness = []
    losses = []
    CeLoss = torch.nn.CrossEntropyLoss()
    batches = data_loader
    max_count = len(batches)
    if not full:
        max_count = 100
    cur_count = 0
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
            losses.append(loss)
            correctness.append(pred == target)
        cur_count += 1
        if cur_count == max_count:
            break
    loss = sum(losses)/max_count
    acc = torch.cat(correctness).to(torch.float).mean()
    logger.info(f'Validation Acc = {acc}, Loss = {loss}')


def validate_sincnet(hparams, dataset, model, progress, full=True):
    logger.info('Performing Valitation')
    accuracies = []
    num_matches = 0
    CeLoss = torch.nn.NLLLoss()
    batches = dataset

    total_loss = 0.0

    max_count = len(batches)
    if not full:
        max_count = 100

    for i, batch in enumerate(batches):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = torch_utils.to_var(v)
        if model.num_classes == 2:
            target = batch['gid']
        else:
            target = batch['cid']
        with torch.no_grad():
            probs = model(batch)
            preds = torch.max(probs, dim=1)[1]
            loss = CeLoss(probs, target)
            val, best_class = torch.max(torch.sum(probs, dim=0), 0)
            num_matches += (best_class == target[0]).float()
            accuracies.append(preds == target)
            total_loss += loss.detach()
        if i > max_count:
            break
    mean_loss = total_loss/max_count
    bmr = num_matches/max_count
    acc = torch.cat(accuracies).to(torch.float).mean()
    logger.info(f'Validation Acc = {acc}, Loss = {mean_loss}, BMR = {bmr}')


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--model_type', default='resnet',
              type=click.Choice(['sincnet', 'resnet']))
@click.option('--resume', is_flag=True, help='Resume trainging from last ckpt')
@click.option('--progress', is_flag=True, help='Show progress bar')
@click.option('--gender', is_flag=True, help='Train Gender Classifier')
@click.option('--ckpt', type=click.Path(exists=True))
@click.option('--duration', default=3.0,
              help=('Duration of samples for training'))
@click.pass_context
def train(ctx, dataset, model_type, resume, progress, gender, ckpt,
          duration):
    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams

    if gender:
        setattr(hparams, 'num_classes', 2)

    file_type = 'raw'

    setattr(app_config, 'progress', progress)

    train_map_file = app_config.map_file[dataset].format(file_type, 'train')
    test_map_file = app_config.map_file[dataset].format(file_type, 'test')

    if model_type == 'resnet':
        dataset = CelebSpeech(train_map_file, tdur=duration)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=hparams.num_workers
        )

        val_dataset = CelebSpeech(test_map_file)

        model = SpecNet(hparams.num_classes, hparams.sf,
                        hparams.win_size, hparams.hop_len)
        validator = partial(validate, hparams, val_dataset, model, progress)
        optimizer_name = 'sgd'
    else:
        dataset = RawSpeech(train_map_file, hparams.duration)
        val_dataset = RawSpeechChunks(test_map_file, hparams.duration,
                                      hparams.overlap, hparams.batch_size)
        model = SpeechNet(hparams)
        validator = partial(validate_sincnet, hparams, val_dataset, model,
                            progress)
        data_loader = DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=app_config.num_workers
        )
        optimizer_name = 'rmsprop'

    model.to(torch_utils.device)

    trainer = training_utils.Trainer(hparams, app_config, model)

    if resume:
        trainer.load_checkpoint(ckpt)

    params = hparams.optimizer[optimizer_name]['params']
    trainer.setup_optimizers(optimizer_name, params, resume)

    for epoch in range(hparams.epochs):
        trainer.train(dataset, hparams.num_workers, data_loader,
                      validator=validator)
        trainer.save_checkpoint()
