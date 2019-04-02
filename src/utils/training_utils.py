#!/usr/bin/env python
import logging
import os
import torch
import math
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from src.utils import torch_utils


logger = logging.getLogger(__name__)


class LogarithmicDecay(object):
    def __init__(self, init_lr, fin_lr, last_epoch):
        self.init_lr = init_lr
        self.fin_lr = fin_lr
        self.last_epoch = last_epoch

    def __call__(self, epoch):
        fact = (self.fin_lr - self.init_lr)/math.log(self.last_epoch)
        lr = fact*math.log(epoch, 2) + self.init_lr
        return lr


class Trainer(object):
    def __init__(self, hparams, app_config, model: nn.Module):
        self.last_epoch = hparams.epochs
        self.save_format = app_config.save_format
        self.model = model
        self.batch_size = hparams.batch_size
        self.pbar = app_config.progress
        self._create_save_dir(app_config.save_path, app_config.save_format)
        self.val_step = hparams.val_step
        val_start = hparams.val_start
        self.val_ofs = self.val_step - val_start

    def _create_save_dir(self, save_path, save_format):
        self.save_path = save_path.format(type(self.model).__name__)
        curr_time = datetime.now()
        ts = curr_time.strftime(save_format)
        self.save_model_path = os.path.join(self.save_path, ts)
        os.makedirs(self.save_model_path)

    def _setup_adam(self, params):
        lr = params['learning_rate']
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'bn', 'downsample.1']

        optimizer_grouped_parameters = [
                {
                    'params': [
                        p for n, p in param_optimizer if not any(
                            nd in n for nd in no_decay)],
                    'weight_decay': params['weight_decay']},
                {
                    'params': [p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay)],
                    'weight_decay': 0.0}
                ]
        self.optimizer = optim.Adam(optimizer_grouped_parameters, lr=lr,
                                    amsgrad=True)
        self.scheduler = LambdaLR(self.optimizer, lambda x: x)

    def _setup_rmsprop(self, params):
        lr = params['learning_rate']
        alpha = params['alpha']
        weight_decay = params['weight_decay']
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=lr, alpha=alpha,
            weight_decay=weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lambda x: x)

    def _setup_sgd(self, params):
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'bn', 'downsample.1']

        optimizer_grouped_parameters = [
                {
                    'params': [
                        p for n, p in param_optimizer if not any(
                            nd in n for nd in no_decay)],
                    'initial_lr': params['init_lr'],
                    'weight_decay': params['weight_decay']},
                {
                    'params': [p for n, p in param_optimizer if any(
                        nd in n for nd in no_decay)],
                    'initial_lr': params['init_lr'],
                    'weight_decay': 0.0}
                ]
        self.optimizer = optim.SGD(
            optimizer_grouped_parameters, lr=params['init_lr'],
            momentum=0.9)
        decay = LogarithmicDecay(
            params['init_lr'], params['fin_lr'], params['last_epoch'])
        self.scheduler = LambdaLR(self.optimizer, decay, params['last_epoch'])

    def setup_optimizers(self, optimizer, params, resume: bool):
        if optimizer == 'adam':
            self._setup_adam(params)
        elif optimizer == 'sgd':
            self._setup_sgd(params)
        elif optimizer == 'rmsprop':
            self._setup_rmsprop(params)
        self.epoch = 0

        if resume:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            self.epoch = self.checkpoint['epoch']

    def train(self, dataset: Dataset, num_workers: int,
              data_loader: DataLoader, validator=None):
        self.epoch += 1
        self.scheduler.step()
        batches = data_loader

        if self.pbar:
            batches = tqdm(batches)

        metric = self.model.loss_obj()

        for step, batch in enumerate(batches):
            if (step + self.val_ofs) % self.val_step == 0:
                self.validate(validator)
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = torch_utils.to_var(v)

            model_outs = self.model(batch)
            upd, total = self.model.loss(model_outs, batch)

            self.optimizer.zero_grad()
            total.backward()

            self.optimizer.step()

            metric.update(upd)
            if self.pbar:
                batches.set_postfix(metric.pbardict())

        self.validate(validator, epoch_done=True)

        logger.info(f'Epoch {self.epoch} done')

    def validate(self, validator, epoch_done=False):
        self.model.eval()
        if epoch_done and (self.epoch + 1) % 8 == 0:
            validator()
        else:
            validator(full=False)

    def save_checkpoint(self):
        epoch_file = 'E{}.'.format(self.epoch) + type(self.model).__name__
        epoch_path = os.path.join(self.save_model_path, epoch_file)
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path=None):
        if path is not None:
            self.checkpoint = torch.load(path)
            self.model.load_state_dict(self.checkpoint['state_dict'])
            self.model.train()
            return
        model_paths = [f.path for f in os.scandir(
            self.save_path) if f.is_dir()]
        model_paths = [os.path.normpath(path) for path in model_paths]
        model_paths = [os.path.basename(path) for path in model_paths]
        model_times = [datetime.strptime(
            mp, self.save_format) for mp in model_paths]
        latest_model = os.path.join(self.save_path, str(max(model_times)))
        extension = type(self.model).__name__
        all_epochs = []
        for epoch in os.listdir(latest_model):
            if epoch.endswith(extension):
                all_epochs.append(os.path.join(latest_model, epoch))

        latest_epoch = max(all_epochs)
        logger.info('Loading checkpoint {}'.format(latest_epoch))
        self.checkpoint = torch.load(latest_epoch)
        self.model.load_state_dict(self.checkpoint['state_dict'])
