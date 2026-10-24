"""Outils d'entraînement et de prédiction du forecaster MTST."""

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from mtst.Loss.Custom_loss import get_loss


def resolve_device(device="auto"):
    # Priorité au GPU disponible, sinon CPU.
    if device != "auto":
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA n'est pas disponible")
        if selected.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS n'est pas disponible")
        return selected
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tensor(value, name):
    # Les tenseurs sont float32 pour éviter les incompatibilités CPU/GPU.
    result = torch.as_tensor(value, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} contient des valeurs non finies")
    return result


def _dataset(model, X, y, mask=None):
    X = _tensor(X, "X")
    y = _tensor(y, "y")
    if X.ndim != 3 or X.shape[-1] != model.input_dim:
        raise ValueError(f"X attendu : (N, seq_len, {model.input_dim})")
    if tuple(y.shape) != (len(X), *model.output_shape):
        raise ValueError(f"y attendu : (N, {', '.join(map(str, model.output_shape))})")
    if len(X) == 0:
        raise ValueError("Dataset vide")
    if mask is None:
        mask = torch.ones_like(y)
    else:
        mask = _tensor(mask, "mask")
        if mask.shape != y.shape:
            raise ValueError("mask doit avoir la même forme que y")
        if (mask < 0).any():
            raise ValueError("mask doit être positif ou nul")
    return TensorDataset(X.cpu(), y.cpu(), mask.cpu())


def fit_model(model, X, y, *, mask=None, epochs=30, batch_size=32,
              lr=1e-3, loss="mse", loss_kwargs=None, device="auto",
              shuffle=False, weight_decay=1e-5, verbose=True):
    """Entraîne sur des données déjà préparées. Retourne la liste des losses."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs et batch_size doivent être positifs")
    dataset = _dataset(model, X, y, mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    target_device = resolve_device(device)
    model.to(target_device)
    criterion = get_loss(loss, **(loss_kwargs or {})).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(epochs):
        model.train()
        weighted_loss, valid_count = 0.0, 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(target_device), yb.to(target_device), mb.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            batch_loss = criterion(pred, yb, mb)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError("Loss non finie")
            batch_loss.backward()
            optimizer.step()
            count = mb.gt(0).sum().item()
            weighted_loss += batch_loss.detach().item() * count
            valid_count += count
        epoch_loss = weighted_loss / max(valid_count, 1)
        history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {epoch_loss:.6f}")
    return history


def predict_model(model, X, *, batch_size=256, device="auto"):
    """Prédit sans gradients. Retourne un tenseur CPU."""
    if batch_size < 1:
        raise ValueError("batch_size doit être positif")
    X = _tensor(X, "X")
    if X.ndim != 3 or X.shape[-1] != model.input_dim:
        raise ValueError(f"X attendu : (N, seq_len, {model.input_dim})")
    if len(X) == 0:
        return torch.empty((0, *model.output_shape))
    target_device = resolve_device(device)
    model.to(target_device)
    was_training = model.training
    model.eval()
    result = []
    try:
        with torch.inference_mode():
            for (xb,) in DataLoader(TensorDataset(X.cpu()), batch_size=batch_size):
                result.append(model(xb.to(target_device)).cpu())
    finally:
        model.train(was_training)
    return torch.cat(result, dim=0)


def save_model(model, path):
    """Sauvegarde les poids et les hyperparamètres (pas les scalers utilisateur)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": copy.deepcopy(model.config), "state_dict": model.state_dict()}, path)


def load_model(cls, path, *, device="auto"):
    """Charge uniquement un checkpoint créé par cette bibliothèque et de confiance."""
    target_device = resolve_device(device)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = cls(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(target_device)
