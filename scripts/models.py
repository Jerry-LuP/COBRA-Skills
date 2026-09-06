from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class MLPConfig:
    hidden_dim: int = 64
    epochs: int = 50
    lr: float = 1e-3
    l2: float = 1e-4
    min_samples: int = 3
    seed: int = 0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    verbose: bool = False
    log_prefix: str = "[mlp]"


_ModuleBase = nn.Module if nn is not None else object


class _TorchMLP(_ModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for _TorchMLP but is not installed.")
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward_with_hidden(self, x: Any) -> tuple[Any, Any]:
        hidden_pre = self.fc1(x)
        hidden = torch.relu(hidden_pre)
        out = self.fc2(hidden)
        return out, hidden_pre

    def forward(self, x: Any) -> Any:
        out, _ = self.forward_with_hidden(x)
        return out


class TorchMLPRegressor:
    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for TorchMLPRegressor but is not installed.")
        self.input_dim = input_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.model = _TorchMLP(input_dim, config.hidden_dim).to(self.device)
        self.trained = False
        self.prior = 0.5
        self.last_fit_diagnostics: dict[str, Any] = {
            "n_samples": 0,
            "trained": False,
            "reason": "not_fit",
        }

    def _loss_from_tensors(self, pred: Any, y: Any) -> float:
        mse = float(torch.mean((pred - y) ** 2).item())
        reg = self._regularization_loss()
        return mse + reg

    def _regularization_loss(self) -> float:
        reg = 0.0
        for name, param in self.model.named_parameters():
            if param.ndim > 1:
                reg += float(torch.sum(param * param).item())
        return 0.5 * self.config.l2 * reg

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if len(y) == 0:
            self.trained = False
            self.prior = 0.5
            self.last_fit_diagnostics = {
                "n_samples": 0,
                "trained": False,
                "reason": "no_samples",
                "prior": self.prior,
            }
            if self.config.verbose:
                print(
                    f"{self.config.log_prefix} skip train n=0 reason=no_samples prior={self.prior:.6f}",
                    flush=True,
                )
            return self.last_fit_diagnostics
        y2 = y.astype(np.float64).reshape(-1, 1)
        self.prior = float(np.mean(y2))
        if len(y2) < self.config.min_samples:
            self.trained = False
            pred = np.full_like(y2, self.prior)
            self.last_fit_diagnostics = {
                "n_samples": int(len(y2)),
                "trained": False,
                "reason": "below_min_samples",
                "prior": self.prior,
                "train_mse": float(np.mean((pred - y2) ** 2)),
                "target_mean": self.prior,
                "target_std": float(np.std(y2)),
            }
            if self.config.verbose:
                print(
                    f"{self.config.log_prefix} skip train n={len(y2)} reason=below_min_samples "
                    f"prior={self.prior:.6f} train_mse={float(np.mean((pred - y2) ** 2)):.6f}",
                    flush=True,
                )
            return self.last_fit_diagnostics
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y2, dtype=torch.float32, device=self.device)
        self.model.train()
        with torch.no_grad():
            pred0, hidden_pre0 = self.model.forward_with_hidden(x_t)
            initial_mse = float(torch.mean((pred0 - y_t) ** 2).item())
            initial_loss = self._loss_from_tensors(pred0, y_t)
            initial_active_relu_fraction = float(torch.mean((hidden_pre0 > 0).float()).item())
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            weight_decay=self.config.l2,
        )
        epoch_losses: list[float] = []
        log_every = max(1, int(np.ceil(self.config.epochs * 0.05)))
        for epoch_idx in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            pred, _ = self.model.forward_with_hidden(x_t)
            loss = torch.mean((pred - y_t) ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred_after, _ = self.model.forward_with_hidden(x_t)
                epoch_loss = self._loss_from_tensors(pred_after, y_t)
                epoch_losses.append(epoch_loss)
            if self.config.verbose and (
                epoch_idx == 0
                or (epoch_idx + 1) % log_every == 0
                or epoch_idx + 1 == self.config.epochs
            ):
                print(
                    f"{self.config.log_prefix} epoch={epoch_idx + 1:03d}/{self.config.epochs:03d} "
                    f"loss={epoch_loss:.6f}",
                    flush=True,
                )
        self.trained = True
        self.model.eval()
        with torch.no_grad():
            final_pred_t, final_hidden_pre = self.model.forward_with_hidden(x_t)
            final_pred = final_pred_t.detach().cpu().numpy().reshape(-1)
            final_hidden_pre_np = final_hidden_pre.detach().cpu().numpy()
        self.last_fit_diagnostics = {
            "n_samples": int(len(y2)),
            "trained": True,
            "reason": "fit",
            "prior": self.prior,
            "epochs": int(self.config.epochs),
            "lr": float(self.config.lr),
            "l2": float(self.config.l2),
            "optimizer": "adam",
            "adam_beta1": float(self.config.adam_beta1),
            "adam_beta2": float(self.config.adam_beta2),
            "adam_eps": float(self.config.adam_eps),
            "initial_loss": float(initial_loss),
            "final_loss": float(epoch_losses[-1] if epoch_losses else initial_loss),
            "initial_train_mse": initial_mse,
            "final_train_mse": float(np.mean((final_pred.reshape(-1, 1) - y2) ** 2)),
            "target_mean": self.prior,
            "target_std": float(np.std(y2)),
            "pred_mean": float(np.mean(final_pred)),
            "pred_std": float(np.std(final_pred)),
            "pred_min": float(np.min(final_pred)),
            "pred_max": float(np.max(final_pred)),
            "active_relu_fraction": float(np.mean(final_hidden_pre_np > 0.0)),
            "initial_active_relu_fraction": initial_active_relu_fraction,
            "epoch_losses": [float(value) for value in epoch_losses],
        }
        return self.last_fit_diagnostics

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if not self.trained:
            return np.full(x.shape[0], self.prior, dtype=np.float64)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_t).reshape(-1).detach().cpu().numpy().astype(np.float64)
        return np.clip(pred, 0.0, 1.0)

    def state_dict(self) -> dict[str, Any]:
        params = {name: tensor.detach().cpu().tolist() for name, tensor in self.model.state_dict().items()}
        return {
            "trained": self.trained,
            "prior": self.prior,
            "config": self.config.__dict__,
            "device": str(self.device),
            "model_state_dict": params,
            "last_fit_diagnostics": self.last_fit_diagnostics,
        }


class MeanRegressor:
    def __init__(self, input_dim: int, config: MLPConfig) -> None:
        self.input_dim = input_dim
        self.config = config
        self.trained = False
        self.prior = 0.5
        self.last_fit_diagnostics: dict[str, Any] = {
            "n_samples": 0,
            "trained": False,
            "reason": "not_fit",
            "backend": "numpy_mean",
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if len(y) == 0:
            self.prior = 0.5
            self.trained = False
            reason = "no_samples"
        else:
            self.prior = float(np.mean(y))
            self.trained = True
            reason = "fit_mean_fallback"
        pred = np.full(len(y), self.prior, dtype=np.float64)
        self.last_fit_diagnostics = {
            "n_samples": int(len(y)),
            "trained": self.trained,
            "reason": reason,
            "prior": self.prior,
            "train_mse": float(np.mean((pred - y) ** 2)) if len(y) else 0.0,
            "target_mean": float(np.mean(y)) if len(y) else 0.0,
            "target_std": float(np.std(y)) if len(y) else 0.0,
            "backend": "numpy_mean",
        }
        if self.config.verbose:
            print(
                f"{self.config.log_prefix} numpy_mean n={len(y)} reason={reason} prior={self.prior:.6f}",
                flush=True,
            )
        return self.last_fit_diagnostics

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.full(x.shape[0], self.prior, dtype=np.float64)

    def state_dict(self) -> dict[str, Any]:
        return {
            "trained": self.trained,
            "prior": self.prior,
            "config": self.config.__dict__,
            "last_fit_diagnostics": self.last_fit_diagnostics,
            "backend": "numpy_mean",
        }


NumpyMLPRegressor = TorchMLPRegressor if torch is not None and nn is not None else MeanRegressor


class LinearUCB:
    def __init__(self, dim: int, *, nu: float = 0.3, lambda_: float = 0.1) -> None:
        self.dim = dim
        self.nu = nu
        self.lambda_ = lambda_
        self._history = np.empty((0, dim), dtype=np.float64)
        self._dual_inv = np.empty((0, 0), dtype=np.float64)
        self._inverse_dirty = False
        self.num_updates = 0

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(1, -1)
        if z.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {z.shape[1]}")
        self._history = np.concatenate((self._history, z), axis=0)
        self._inverse_dirty = True
        self.num_updates += 1

    def _ensure_inverse(self) -> None:
        if self._inverse_dirty:
            gram = self._history @ self._history.T
            gram.flat[:: len(gram) + 1] += self.lambda_
            self._dual_inv = np.linalg.inv(gram)
            self._inverse_dirty = False

    def fit(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[1]}")
        self._history = x.copy()
        if len(x):
            self._inverse_dirty = True
            self._ensure_inverse()
        else:
            self._dual_inv = np.empty((0, 0), dtype=np.float64)
            self._inverse_dirty = False
        self.num_updates = int(len(x))

    def bonus(self, z: np.ndarray) -> float:
        return float(self.bonus_many(np.asarray(z, dtype=np.float64).reshape(1, -1))[0])

    def bonus_many(self, x: np.ndarray) -> np.ndarray:
        self._ensure_inverse()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[1]}")
        squared_norms = np.einsum("ij,ij->i", x, x, optimize=True)
        if len(self._history):
            projections = x @ self._history.T
            corrections = np.einsum(
                "ij,jk,ik->i",
                projections,
                self._dual_inv,
                projections,
                optimize=True,
            )
            variances = (squared_norms - corrections) / self.lambda_
        else:
            variances = squared_norms / self.lambda_
        return self.nu * np.sqrt(np.maximum(variances, 0.0))

    def state_dict(self) -> dict[str, Any]:
        self._ensure_inverse()
        return {
            "dim": self.dim,
            "nu": self.nu,
            "lambda": self.lambda_,
            "num_updates": self.num_updates,
            "representation": "dual",
            "history_X": self._history.tolist(),
            "dual_inverse": self._dual_inv.tolist(),
        }
