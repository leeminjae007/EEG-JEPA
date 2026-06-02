import copy
import math
import os
import sys
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(params.device)

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.to(self.device)
        self.device_ids = list(getattr(self.params, 'device_ids', []))
        self.use_data_parallel = (
            self.device.type == 'cuda'
            and bool(getattr(self.params, 'data_parallel', False))
            and len(self.device_ids) > 1
        )
        if self.use_data_parallel:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0],
            )
            self._print(f'Using DataParallel on GPUs {self.device_ids}')

        if self.params.task_type == 'multiclass':
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).to(self.device)
        elif self.params.task_type == 'regression':
            self.criterion = MSELoss().to(self.device)
        else:
            self.criterion = BCEWithLogitsLoss().to(self.device)

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
                param.requires_grad = not self.params.frozen
            else:
                other_params.append(param)

        if self.params.optimizer == 'SGD':
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5.0},
                ], momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.params.lr,
                    momentum=0.9,
                    weight_decay=self.params.weight_decay,
                )
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': 0.001 * (self.params.batch_size / 256.0) ** 0.5},
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.params.lr,
                    weight_decay=self.params.weight_decay,
                )

        self.data_length = len(self.data_loader['train'])
        self.total_steps = max(int(self.params.epochs) * max(self.data_length, 1), 1)
        self.warmup_steps = max(int(getattr(self.params, 'warmup_epochs', 0)) * max(self.data_length, 1), 0)
        self.min_lr = float(getattr(self.params, 'min_lr', 1e-6))
        self.base_lrs = [float(group['lr']) for group in self.optimizer.param_groups]
        self.global_step = 0
        self.max_train_batches = max(int(getattr(self.params, 'max_train_batches', 0)), 0)
        self.log_interval_steps = max(int(getattr(self.params, 'log_interval_steps', 20)), 0)
        self.show_tqdm = bool(getattr(self.params, 'show_tqdm', sys.stderr.isatty()))
        self.wandb_run = self._init_wandb()
        self._set_learning_rate(self.global_step)
        self._print(self.model)

    def _lr_at_step(self, base_lr, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * float(step + 1) / float(self.warmup_steps)
        cosine_steps = max(self.total_steps - self.warmup_steps, 1)
        progress = float(step - self.warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_learning_rate(self, step):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = self._lr_at_step(base_lr, step)

    def _init_wandb(self):
        if not getattr(self.params, 'wandb', False):
            return None
        if wandb is None:
            self._print('wandb is not installed; run `pip install wandb` to enable W&B logging')
            return None
        return wandb.init(
            project=getattr(self.params, 'wandb_project', 'eeg-jepa-finetune'),
            entity=getattr(self.params, 'wandb_entity', None),
            name=getattr(self.params, 'wandb_name', None),
            group=getattr(self.params, 'wandb_group', None),
            tags=getattr(self.params, 'wandb_tags', None),
            dir=getattr(self.params, 'model_dir', None),
            config=vars(self.params),
            resume='allow',
        )

    def _wandb_log(self, metrics, step=None):
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def _finish_wandb(self):
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None

    def _save_best(self, model_path):
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        torch.save(self._model_state_dict(), model_path)
        self._print('model save in ' + model_path)

    def _print(self, message):
        print(message, flush=True)

    def _raw_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def _model_state_dict(self):
        return self._raw_model().state_dict()

    def _load_model_state_dict(self, state_dict):
        self._raw_model().load_state_dict(state_dict)

    def _train_total_steps(self):
        if self.max_train_batches > 0:
            return min(self.data_length, self.max_train_batches)
        return self.data_length

    def _train_iterator(self):
        return tqdm(
            self.data_loader['train'],
            mininterval=10,
            dynamic_ncols=True,
            disable=not self.show_tqdm,
        )

    def _log_step_progress(self, epoch_idx, batch_idx, loss_value, start_time):
        if self.log_interval_steps <= 0:
            return
        step_in_epoch = batch_idx + 1
        total_steps = self._train_total_steps()
        should_log = (
            step_in_epoch == 1
            or step_in_epoch % self.log_interval_steps == 0
            or step_in_epoch == total_steps
        )
        if not should_log:
            return
        lr = self.optimizer.param_groups[0]['lr']
        elapsed_min = (timer() - start_time) / 60.0
        self._print(
            'Epoch {}/{} Step {}/{} loss={:.5f} lr={:.6g} elapsed={:.2f}m'.format(
                epoch_idx + 1,
                self.params.epochs,
                step_in_epoch,
                total_steps,
                loss_value,
                lr,
                elapsed_min,
            )
        )

    def train_for_multiclass(self):
        best_score = -1.0
        best_epoch = 0
        try:
            for epoch in range(self.params.epochs):
                self.model.train()
                start_time = timer()
                losses = []

                for batch_idx, (x, y) in enumerate(self._train_iterator()):
                    if self.max_train_batches > 0 and batch_idx >= self.max_train_batches:
                        break
                    self.optimizer.zero_grad()
                    self._set_learning_rate(self.global_step)
                    x = x.to(self.device)
                    y = y.to(self.device).long()
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    loss_value = loss.detach().cpu().item()
                    losses.append(loss_value)

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                    self.optimizer.step()
                    self.global_step += 1
                    self._log_step_progress(epoch, batch_idx, loss_value, start_time)
                    self._wandb_log({'train/loss_step': loss_value, 'train/epoch': epoch + 1}, step=self.global_step)

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                with torch.no_grad():
                    acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                    train_loss = float(np.mean(losses)) if losses else 0.0
                    elapsed_min = (timer() - start_time) / 60
                    self._print(
                        'Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                            epoch + 1,
                            train_loss,
                            acc,
                            kappa,
                            f1,
                            lr,
                            elapsed_min,
                        )
                    )
                    self._print(cm)
                    self._wandb_log(
                        {
                            'val/acc': acc,
                            'val/kappa': kappa,
                            'val/f1': f1,
                            'train/loss_epoch': train_loss,
                            'train/lr': lr,
                            'time/epoch_minutes': elapsed_min,
                            'epoch': epoch + 1,
                        },
                        step=self.global_step,
                    )

                    if kappa > best_score:
                        best_score = kappa
                        best_epoch = epoch + 1
                        self.best_model_states = copy.deepcopy(self._model_state_dict())

            self._load_model_state_dict(self.best_model_states)
            with torch.no_grad():
                acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
                self._print('***************************Test results************************')
                self._print('Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}'.format(acc, kappa, f1))
                self._print(cm)
                self._wandb_log(
                    {
                        'test/acc': acc,
                        'test/kappa': kappa,
                        'test/f1': f1,
                        'best/epoch': best_epoch,
                    },
                    step=self.global_step,
                )
                self._save_best(
                    os.path.join(self.params.model_dir, f'epoch{best_epoch}_acc_{acc:.5f}_kappa_{kappa:.5f}_f1_{f1:.5f}.pth')
                )
        finally:
            self._finish_wandb()

    def train_for_binaryclass(self):
        best_score = -1.0
        best_epoch = 0
        try:
            for epoch in range(self.params.epochs):
                self.model.train()
                start_time = timer()
                losses = []

                for batch_idx, (x, y) in enumerate(self._train_iterator()):
                    if self.max_train_batches > 0 and batch_idx >= self.max_train_batches:
                        break
                    self.optimizer.zero_grad()
                    self._set_learning_rate(self.global_step)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    loss_value = loss.detach().cpu().item()
                    losses.append(loss_value)

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                    self.optimizer.step()
                    self.global_step += 1
                    self._log_step_progress(epoch, batch_idx, loss_value, start_time)
                    self._wandb_log({'train/loss_step': loss_value, 'train/epoch': epoch + 1}, step=self.global_step)

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                with torch.no_grad():
                    acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                    train_loss = float(np.mean(losses)) if losses else 0.0
                    elapsed_min = (timer() - start_time) / 60
                    self._print(
                        'Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                            epoch + 1,
                            train_loss,
                            acc,
                            pr_auc,
                            roc_auc,
                            lr,
                            elapsed_min,
                        )
                    )
                    self._print(cm)
                    self._wandb_log(
                        {
                            'val/acc': acc,
                            'val/pr_auc': pr_auc,
                            'val/roc_auc': roc_auc,
                            'train/loss_epoch': train_loss,
                            'train/lr': lr,
                            'time/epoch_minutes': elapsed_min,
                            'epoch': epoch + 1,
                        },
                        step=self.global_step,
                    )

                    if roc_auc > best_score:
                        best_score = roc_auc
                        best_epoch = epoch + 1
                        self.best_model_states = copy.deepcopy(self._model_state_dict())

            self._load_model_state_dict(self.best_model_states)
            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
                self._print('***************************Test results************************')
                self._print('Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}'.format(acc, pr_auc, roc_auc))
                self._print(cm)
                self._wandb_log(
                    {
                        'test/acc': acc,
                        'test/pr_auc': pr_auc,
                        'test/roc_auc': roc_auc,
                        'best/epoch': best_epoch,
                    },
                    step=self.global_step,
                )
                self._save_best(
                    os.path.join(self.params.model_dir, f'epoch{best_epoch}_acc_{acc:.5f}_pr_{pr_auc:.5f}_roc_{roc_auc:.5f}.pth')
                )
        finally:
            self._finish_wandb()

    def train_for_regression(self):
        best_score = -1.0
        best_epoch = 0
        try:
            for epoch in range(self.params.epochs):
                self.model.train()
                start_time = timer()
                losses = []

                for batch_idx, (x, y) in enumerate(self._train_iterator()):
                    if self.max_train_batches > 0 and batch_idx >= self.max_train_batches:
                        break
                    self.optimizer.zero_grad()
                    self._set_learning_rate(self.global_step)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    loss_value = loss.detach().cpu().item()
                    losses.append(loss_value)

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                    self.optimizer.step()
                    self.global_step += 1
                    self._log_step_progress(epoch, batch_idx, loss_value, start_time)
                    self._wandb_log({'train/loss_step': loss_value, 'train/epoch': epoch + 1}, step=self.global_step)

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                with torch.no_grad():
                    corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                    train_loss = float(np.mean(losses)) if losses else 0.0
                    elapsed_min = (timer() - start_time) / 60
                    self._print(
                        'Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                            epoch + 1,
                            train_loss,
                            corrcoef,
                            r2,
                            rmse,
                            lr,
                            elapsed_min,
                        )
                    )
                    self._wandb_log(
                        {
                            'val/corrcoef': corrcoef,
                            'val/r2': r2,
                            'val/rmse': rmse,
                            'train/loss_epoch': train_loss,
                            'train/lr': lr,
                            'time/epoch_minutes': elapsed_min,
                            'epoch': epoch + 1,
                        },
                        step=self.global_step,
                    )

                    if r2 > best_score:
                        best_score = r2
                        best_epoch = epoch + 1
                        self.best_model_states = copy.deepcopy(self._model_state_dict())

            self._load_model_state_dict(self.best_model_states)
            with torch.no_grad():
                corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
                self._print('***************************Test results************************')
                self._print('Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}'.format(corrcoef, r2, rmse))
                self._wandb_log(
                    {
                        'test/corrcoef': corrcoef,
                        'test/r2': r2,
                        'test/rmse': rmse,
                        'best/epoch': best_epoch,
                    },
                    step=self.global_step,
                )
                self._save_best(
                    os.path.join(self.params.model_dir, f'epoch{best_epoch}_corrcoef_{corrcoef:.5f}_r2_{r2:.5f}_rmse_{rmse:.5f}.pth')
                )
        finally:
            self._finish_wandb()
