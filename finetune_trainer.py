import copy
import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator


class Trainer:
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(params.device)

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.to(self.device)

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
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.params.epochs * self.data_length,
            eta_min=1e-6,
        )
        print(self.model)

    def _save_best(self, model_path):
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)
        torch.save(self.model.state_dict(), model_path)
        print('model save in ' + model_path)

    def train_for_multiclass(self):
        best_score = -1.0
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device).long()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                losses.append(loss.detach().cpu().item())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    'Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        lr,
                        (timer() - start_time) / 60,
                    )
                )
                print(cm)

                if kappa > best_score:
                    best_score = kappa
                    best_epoch = epoch + 1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print('***************************Test results************************')
            print('Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}'.format(acc, kappa, f1))
            print(cm)
            self._save_best(
                os.path.join(self.params.model_dir, f'epoch{best_epoch}_acc_{acc:.5f}_kappa_{kappa:.5f}_f1_{f1:.5f}.pth')
            )

    def train_for_binaryclass(self):
        best_score = -1.0
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                losses.append(loss.detach().cpu().item())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    'Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        pr_auc,
                        roc_auc,
                        lr,
                        (timer() - start_time) / 60,
                    )
                )
                print(cm)

                if roc_auc > best_score:
                    best_score = roc_auc
                    best_epoch = epoch + 1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print('***************************Test results************************')
            print('Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}'.format(acc, pr_auc, roc_auc))
            print(cm)
            self._save_best(
                os.path.join(self.params.model_dir, f'epoch{best_epoch}_acc_{acc:.5f}_pr_{pr_auc:.5f}_roc_{roc_auc:.5f}.pth')
            )

    def train_for_regression(self):
        best_score = -1.0
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []

            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                losses.append(loss.detach().cpu().item())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    'Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins'.format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        lr,
                        (timer() - start_time) / 60,
                    )
                )

                if r2 > best_score:
                    best_score = r2
                    best_epoch = epoch + 1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print('***************************Test results************************')
            print('Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}'.format(corrcoef, r2, rmse))
            self._save_best(
                os.path.join(self.params.model_dir, f'epoch{best_epoch}_corrcoef_{corrcoef:.5f}_r2_{r2:.5f}_rmse_{rmse:.5f}.pth')
            )
