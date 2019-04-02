#!/usr/bin/env python
import logging
import os
import torch
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from tqdm import tqdm
from src.utils import torch_utils


logger = logging.getLogger(__name__)


class InterpolatingScheduler(_LRScheduler):
    def __init__(self, optimizer, steps, lrs, scale='log', last_epoch=-1):
        """A scheduler that interpolates given values

        Args:
        - optimizer: pytorch optimizer
        - steps: list or array with the x coordinates of the values
        - lrs: list or array with the learning rates corresponding to the steps
        - scale: one of ['linear', 'log'] the scale on which to interpolate.
                 Log is usefull since learning rates operate on a
                 logarithmic scale.

        Usage:
            fc = nn.Linear(1,1)
            optimizer = optim.Adam(fc.parameters())
            lr_scheduler = InterpolatingScheduler(optimizer,
            steps=[0, 100, 400], lrs=[1e-6, 1e-4, 1e-8], scale='log')
        """
        self.scale = scale
        self.steps = steps
        self.lrs = lrs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        x = [self.last_epoch]
        if self.scale == 'linear':
            y = np.interp(x, self.steps, self.lrs)
        elif self.scale == 'log':
            y = np.interp(x, self.steps, np.log(self.lrs))
            y = np.exp(y)
        else:
            raise ValueError("scale should be one of ['linear', 'log']")
        logger.debug(f'Epoch = {self.last_epoch}, lr = {y[0]}')
        return [y[0] for lr in self.base_lrs]


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

    def get_filter_parameters(self, params: dict):
        no_decay = params.get('decay_filters')
        if no_decay is None:
            return self.model.parameters()
        param_optimizer = list(self.model.named_parameters())

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
        return optimizer_grouped_parameters

    def _create_save_dir(self, save_path, save_format):
        self.save_path = save_path.format(type(self.model).__name__)
        curr_time = datetime.now()
        ts = curr_time.strftime(save_format)
        self.save_model_path = os.path.join(self.save_path, ts)
        os.makedirs(self.save_model_path)

    def _setup_adam(self, params):
        lr = params['learning_rate']
        parameters = self.get_filter_parameters(params)
        self.optimizer = optim.Adam(parameters, lr=lr,
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
        parameters = self.get_filter_parameters(params)
        self.optimizer = optim.SGD(
            parameters, lr=params['init_lr'], momentum=0.9)

        self.scheduler = InterpolatingScheduler(
            self.optimizer, [0, params['last_epoch']],
            [params['init_lr'], params['fin_lr']])

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
            assert total < 10.0, f'Step = {step}'
            total.backward()
            assert total < 10.0, f'Step = {step}'

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
