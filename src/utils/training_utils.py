#!/usr/bin/env python
import logging
import os
import torch
from datetime import datetime
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from src.utils import torch_utils


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, hparams, app_config, model: nn.Module):
        self.last_epoch = hparams.epochs
        self.save_format = app_config.save_format
        self.model = model
        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.pbar = app_config.progress
        self._create_save_dir(app_config.save_path, app_config.save_format)
        self.l2_coeff = hparams.l2_coeff
        self.eps = hparams.adam_eps
        self.sched_decay = hparams.sched_decay
        self.val_step = hparams.val_step
        val_start = hparams.val_start
        self.val_ofs = self.val_step - val_start

    def _create_save_dir(self, save_path, save_format):
        self.save_path = save_path.format(type(self.model).__name__)
        curr_time = datetime.now()
        ts = curr_time.strftime(save_format)
        self.save_model_path = os.path.join(self.save_path, ts)
        os.makedirs(self.save_model_path)

    def setup_optimizers(self, resume: bool):
        if resume:
            self.optimizer = self.checkpoint['optimizer']
            self.scheduler = self.checkpoint['scheduler']
            self.epoch = self.checkpoint['epoch']
        else:
            param_optimizer = list(self.model.named_parameters())

            no_decay = ['bias', 'bn', 'downsample.1']

            optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in param_optimizer if not any(
                                nd in n for nd in no_decay)],
                        'weight_decay': self.l2_coeff},
                    {
                        'params': [p for n, p in param_optimizer if any(
                            nd in n for nd in no_decay)], 'weight_decay': 0.0}
                    ]
            self.optimizer = optim.SGD(
                optimizer_grouped_parameters, lr=self.learning_rate,
                momentum=0.9, weight_decay=self.l2_coeff)

            self.scheduler = StepLR(self.optimizer, 1, gamma=self.sched_decay)
            self.epoch = 0

    def train(self, dataset: Dataset, num_workers: int,
              data_loader: DataLoader, validator=None):
        self.epoch += 1
        batches = data_loader

        if self.pbar:
            batches = tqdm(batches)

        metric = self.model.loss_obj()

        for step, batch in enumerate(batches):
            if (step + self.val_ofs) % self.val_step == 0:
                if validator:
                    self.model.eval()
                    validator(full=False)
                    self.model.train()
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = torch_utils.to_var(v)

            model_outs = self.model(batch)
            upd, total = self.model.loss(model_outs, batch)

            self.optimizer.zero_grad()
            total.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            metric.update(upd)
            if self.pbar:
                batches.set_postfix(metric.pbardict())
        self.scheduler.step()
        if validator:
            self.model.eval()
            validator()
            self.model.train()

        logger.info(f'Epoch {self.epoch} done')

    def save_checkpoint(self):
        epoch_file = 'E{}.'.format(self.epoch) + type(self.model).__name__
        epoch_path = os.path.join(self.save_model_path, epoch_file)
        checkpoint = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler}
        torch.save(checkpoint, epoch_path)

    def load_checkpoint(self):
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
