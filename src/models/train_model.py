#!/usr/bin/env python
import click
import torch
import logging
from functools import partial
from torch.utils.data import DataLoader
from src.data.dataset import Spectrogram, RawSpeech, RawSpeechChunks
from src.utils import training_utils
from src.utils import torch_utils
from src.models.resnet import ResnetM
from src.models.speechnet import SpeechNet


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


def validate_sincnet(hparams, dataset, model, progress, full=True):
    logger.info('Performing Valitation')
    accuracies = []
    num_matches = 0
    CeLoss = torch.nn.NLLLoss()
    batches = dataset

    total_loss = 0.0

    max_count = len(batches)
    if not full:
        max_count = 10

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
            loss = CeLoss(preds, target)
            val, best_class = torch.max(torch.sum(probs, dim=0), 0)
            num_matches += (best_class == target[0]).float()
            accuracies.append(preds == target)
            total_loss += loss.detach()
        if i > max_count:
            break
    mean_loss = total_loss/len(batches)
    bmr = num_matches/len(batches)
    acc = torch.cat(accuracies).to(torch.float).mean()
    logger.info(f'Validation Acc = {acc}, Loss = {mean_loss}, BMR = {bmr}')


@click.command()
@click.option('--dataset', default='VoxCeleb1',
              type=click.Choice(['VoxCeleb1', 'VoxCeleb2']))
@click.option('--model_type', default='sincnet',
              type=click.Choice(['sincnet', 'resnet']))
@click.option('--resume', is_flag=True, help='Resume trainging from last ckpt')
@click.option('--progress', is_flag=True, help='Show progress bar')
@click.option('--gender', is_flag=True, help='Train Gender Classifier')
@click.pass_context
def train(ctx, dataset, model_type, resume, progress, gender):
    app_config = ctx.obj.app_config
    hparams = ctx.obj.hparams

    if gender:
        setattr(hparams, 'num_classes', 2)

    file_type = 'raw' if model_type == 'sincnet' else 'sgram'

    setattr(app_config, 'progress', progress)

    train_map_file = app_config.map_file[dataset].format(file_type, 'train')
    test_map_file = app_config.map_file[dataset].format(file_type, 'test')

    if model_type == 'sgram':
        dataset = Spectrogram(train_map_file)

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=hparams.batch_size,
            shuffle=True,
            num_workers=hparams.num_workers
        )

        val_dataset = Spectrogram(test_map_file)

        model = ResnetM(hparams)
        validator = partial(validate, hparams, val_dataset, model, progress)
    else:
        dataset = RawSpeech(train_map_file, hparams.duration)
        val_dataset = RawSpeechChunks(test_map_file, hparams.duration,
                                      hparams.overlap, hparams.batch_size)
        model = SpeechNet(hparams)
        validator = partial(validate_sincnet, hparams, val_dataset, model,
                            progress)

    model.to(torch_utils.device)

    trainer = training_utils.Trainer(hparams, app_config, model)

    if resume:
        trainer.load_checkpoint()

    optimizer = hparams.optimizer['name']
    params = hparams.optimizer['params']
    trainer.setup_optimizers(optimizer, params, resume)

    for epoch in range(hparams.epochs):
        trainer.train(dataset, hparams.num_workers, data_loader,
                      validator=validator)
        trainer.save_checkpoint()
