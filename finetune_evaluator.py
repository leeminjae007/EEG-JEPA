import numpy as np
import torch
from tqdm import tqdm


def _confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if n_classes is None:
        max_true = int(y_true.max()) if y_true.size > 0 else 0
        max_pred = int(y_pred.max()) if y_pred.size > 0 else 0
        n_classes = max(max_true, max_pred) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _balanced_accuracy(y_true, y_pred):
    cm = _confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), row_sum, out=np.zeros_like(row_sum, dtype=np.float64), where=row_sum > 0)
    return float(recall.mean()) if recall.size > 0 else 0.0


def _weighted_f1(y_true, y_pred):
    cm = _confusion_matrix(y_true, y_pred)
    support = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    total = support.sum()
    if total <= 0:
        return 0.0
    return float((f1 * support).sum() / total)


def _cohen_kappa(y_true, y_pred):
    cm = _confusion_matrix(y_true, y_pred)
    n = cm.sum()
    if n <= 0:
        return 0.0
    po = float(np.trace(cm) / n)
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    pe = float((row * col).sum() / (n * n))
    denom = 1.0 - pe
    return float((po - pe) / denom) if denom != 0.0 else 0.0


def _roc_auc_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[pos].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _pr_auc_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1).astype(np.float64)
    fp = np.cumsum(y_true == 0).astype(np.float64)
    n_pos = float((y_true == 1).sum())
    if n_pos <= 0:
        return 0.0
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapezoid(precision, recall))


def _r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    mean_true = np.mean(y_true) if y_true.size > 0 else 0.0
    ss_tot = np.sum((y_true - mean_true) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(params.device)

    def get_metrics_for_multiclass(self, model):
        model.eval()
        truths = []
        preds = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device).long()
            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().reshape(-1).tolist()
            preds += pred_y.cpu().reshape(-1).tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = _balanced_accuracy(truths, preds)
        f1 = _weighted_f1(truths, preds)
        kappa = _cohen_kappa(truths, preds)
        cm = _confusion_matrix(truths, preds)
        return acc, kappa, f1, cm

    def get_metrics_for_binaryclass(self, model):
        model.eval()
        truths = []
        preds = []
        scores = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()

            truths += y.long().cpu().reshape(-1).tolist()
            preds += pred_y.cpu().reshape(-1).tolist()
            scores += score_y.cpu().reshape(-1).tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)

        acc = _balanced_accuracy(truths, preds)
        roc_auc = _roc_auc_binary(truths, scores)
        pr_auc = _pr_auc_binary(truths, scores)
        cm = _confusion_matrix(truths, preds, n_classes=2)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()
        truths = []
        preds = []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.to(self.device)
            y = y.to(self.device)
            pred = model(x)
            truths += y.cpu().reshape(-1).tolist()
            preds += pred.cpu().reshape(-1).tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = _r2_score(truths, preds)
        rmse = float(np.sqrt(np.mean((truths - preds) ** 2)))
        return corrcoef, r2, rmse
