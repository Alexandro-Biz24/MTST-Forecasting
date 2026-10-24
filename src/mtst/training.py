"""Outils d'entraînement et de prédiction du forecaster MTST."""

import copy
import time
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


def _synchronize(device):
    """Synchronize asynchronous accelerator operations for accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rte_load_mape(pred, target, mask, xb, options):
    """Masked MAPE in physical MW, matching the original RTE notebook.

    X and y must contain the *scaled* feature and residual target values.
    Options: feature_index, x_mean, x_scale, y_mean, y_scale.
    """
    required = {"feature_index", "x_mean", "x_scale", "y_mean", "y_scale"}
    missing = required - options.keys()
    if missing:
        raise ValueError(f"rte_mape missing loss_kwargs: {sorted(missing)}")
    index = options["feature_index"]
    if not isinstance(index, int) or not (0 <= index < xb.shape[-1]):
        raise ValueError("feature_index must index an input feature")
    if pred.shape != target.shape or pred.ndim != 2 or pred.shape[1] != xb.shape[1]:
        raise ValueError("rte_mape requires X: (B,T,F) and y/pred: (B,T)")
    baseline = xb[:, :, index] * options["x_scale"] + options["x_mean"]
    prediction_mw = baseline + pred * options["y_scale"] + options["y_mean"]
    target_mw = baseline + target * options["y_scale"] + options["y_mean"]
    return (((prediction_mw - target_mw).abs() / target_mw.abs().clamp(min=1))
            * (mask > 0).to(pred.dtype)).sum() / (mask > 0).sum().clamp(min=1)


def fit_model(model, X, y, *, mask=None, epochs=30, batch_size=32,
              lr=1e-3, loss="mse", loss_kwargs=None, device="auto",
              shuffle=False, weight_decay=1e-5, verbose=True,
              log_every=0):
    """Train on prepared tensors and return one training loss per epoch.

    log_every=0 prints at epoch boundaries. Set log_every=10 to see batch
    progress every 10 batches (including the first and last batch).
    Use loss='rte_mape' with loss_kwargs to match RTE residual training
    in physical MW; returned values are fractions, not percentages.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs et batch_size doivent être positifs")
    if not isinstance(log_every, int) or log_every < 0:
        raise ValueError("log_every must be a nonnegative integer")
    start_total = time.perf_counter()
    if verbose:
        print("Preparing dataset...", flush=True)
    dataset = _dataset(model, X, y, mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    target_device = resolve_device(device)
    model.to(target_device)
    opts = loss_kwargs or {}
    rte_mape = isinstance(loss, str) and loss.lower() == "rte_mape"
    criterion = None if rte_mape else get_loss(loss, **opts).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    if verbose:
        print(f"Training started | loss={loss} | epochs={epochs} | "
              f"batches/epoch={len(loader)} | device={target_device}", flush=True)

    for epoch in range(epochs):
        _synchronize(target_device)
        start_epoch = time.perf_counter()
        model.train()
        weighted_loss, valid_count = 0.0, 0.0
        for batch_index, (xb, yb, mb) in enumerate(loader, 1):
            xb, yb, mb = xb.to(target_device), yb.to(target_device), mb.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            batch_loss = (_rte_load_mape(pred, yb, mb, xb, opts) if rte_mape
                          else criterion(pred, yb, mb))
            if not torch.isfinite(batch_loss).item():
                raise FloatingPointError("Loss non finie")
            batch_loss.backward()
            optimizer.step()
            count = mb.gt(0).sum().item()
            weighted_loss += batch_loss.detach().item() * count
            valid_count += count
            if verbose and log_every and (batch_index == 1 or batch_index % log_every == 0
                                          or batch_index == len(loader)):
                _synchronize(target_device)
                print(f"Epoch {epoch+1:02d}/{epochs} | Batch {batch_index}/{len(loader)} "
                      f"| {str(loss).upper()}: {batch_loss.item():.6f} "
                      f"| elapsed: {time.perf_counter()-start_epoch:.1f}s", flush=True)
        epoch_loss = weighted_loss / max(valid_count, 1)
        history.append(epoch_loss)
        _synchronize(target_device)
        if verbose:
            print(f"Epoch {epoch+1:02d}/{epochs} | {str(loss).upper()}: "
                  f"{epoch_loss:.6f} | time: {time.perf_counter()-start_epoch:.2f}s",
                  flush=True)
    _synchronize(target_device)
    if verbose:
        print(f"Training completed | {str(loss).upper()}: {history[-1]:.6f} "
              f"| total: {(time.perf_counter()-start_total)/60:.2f} min", flush=True)
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
