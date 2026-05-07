"""Small NumPy MLP for candidate grasp success probability."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class MLPBinaryClassifier:
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 32), seed: int = 7):
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        dims = [self.input_dim] + self.hidden_dims + [1]
        rng = np.random.default_rng(seed)
        self.params: Dict[str, np.ndarray] = {}
        for idx in range(len(dims) - 1):
            scale = np.sqrt(2.0 / max(dims[idx], 1))
            self.params[f"W{idx + 1}"] = rng.normal(0.0, scale, size=(dims[idx], dims[idx + 1]))
            self.params[f"b{idx + 1}"] = np.zeros((1, dims[idx + 1]), dtype=np.float64)

    @property
    def num_layers(self) -> int:
        return len(self.hidden_dims) + 1

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        activations = [x]
        out = x
        for layer in range(1, self.num_layers):
            z = out @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            out = np.maximum(z, 0.0)
            activations.append(out)
        logits = out @ self.params[f"W{self.num_layers}"] + self.params[f"b{self.num_layers}"]
        return logits.reshape(-1), activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return sigmoid(logits)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return logits

    def train_batch(self, x: np.ndarray, y: np.ndarray, lr: float, pos_weight: float | None = None) -> float:
        y = y.reshape(-1)
        logits, activations = self.forward(x)
        probs = sigmoid(logits)
        weight = 1.0 if pos_weight is None else float(pos_weight)
        positive_loss = np.logaddexp(0.0, -logits) * y * weight
        negative_loss = np.logaddexp(0.0, logits) * (1.0 - y)
        loss = np.mean(positive_loss + negative_loss)

        grad_logits = np.where(y == 1.0, weight * (probs - 1.0), probs)
        grad = (grad_logits / max(len(y), 1)).reshape(-1, 1)
        grads: Dict[str, np.ndarray] = {}
        for layer in range(self.num_layers, 0, -1):
            prev = activations[layer - 1]
            grads[f"W{layer}"] = prev.T @ grad
            grads[f"b{layer}"] = np.sum(grad, axis=0, keepdims=True)
            if layer > 1:
                grad = grad @ self.params[f"W{layer}"].T
                grad = grad * (prev > 0.0)

        for name, value in grads.items():
            self.params[name] -= lr * value
        return float(loss)

    def state_dict(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "params": self.params,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "MLPBinaryClassifier":
        model = cls(input_dim=int(state["input_dim"]), hidden_dims=state["hidden_dims"])
        model.params = state["params"]
        return model


def save_checkpoint(path: Path, model: MLPBinaryClassifier, scaler: StandardScaler, metadata: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "metadata": metadata,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_checkpoint(path: Path) -> Tuple[MLPBinaryClassifier, StandardScaler, Dict[str, object]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model = MLPBinaryClassifier.from_state_dict(payload["model"])
    scaler_state = payload["scaler"]
    scaler = StandardScaler(mean=scaler_state["mean"], std=scaler_state["std"])
    return model, scaler, payload.get("metadata", {})
