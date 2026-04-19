from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from driver_monitoring_rl.config import DrowsinessTrainConfig


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 40, layers: int = 3, dropout: float = 0.25):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


@dataclass
class TrainResult:
    metrics: Dict[str, float]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


class RNNTrainer:
    def __init__(self, cfg: DrowsinessTrainConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)

    def _fit_one_fold(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[nn.Module, Dict[str, float], StandardScaler]:
        b, t, f = x_train.shape
        scaler = StandardScaler()
        x_train_2d = x_train.reshape(b * t, f)
        scaler.fit(x_train_2d)

        x_train = scaler.transform(x_train_2d).reshape(b, t, f)
        x_val = scaler.transform(x_val.reshape(x_val.shape[0] * t, f)).reshape(x_val.shape[0], t, f)

        train_ds = SequenceDataset(x_train, y_train)
        val_ds = SequenceDataset(x_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        model = SimpleRNNClassifier(
            input_size=f,
            hidden_size=self.cfg.hidden_size,
            layers=self.cfg.num_rnn_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_f1 = -1.0
        best_state = None
        patience = 0

        for _epoch in range(self.cfg.epochs):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics = self._evaluate(model, val_loader)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        metrics = self._evaluate(model, val_loader)
        return model, metrics, scaler

    def _evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, float]:
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).long().cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(yb.long().cpu().numpy().tolist())

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    def cross_validate(self, x: np.ndarray, y: np.ndarray) -> TrainResult:
        skf = StratifiedKFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)
        fold_metrics = []
        scaler_means = []
        scaler_scales = []

        for tr_idx, va_idx in skf.split(x, y):
            _, metrics, scaler = self._fit_one_fold(x[tr_idx], y[tr_idx], x[va_idx], y[va_idx])
            fold_metrics.append(metrics)
            scaler_means.append(scaler.mean_)
            scaler_scales.append(scaler.scale_)

        avg_metrics = {
            key: float(np.mean([m[key] for m in fold_metrics]))
            for key in fold_metrics[0].keys()
        }
        return TrainResult(
            metrics=avg_metrics,
            scaler_mean=np.mean(np.stack(scaler_means), axis=0),
            scaler_scale=np.mean(np.stack(scaler_scales), axis=0),
        )
