"""Fonctions de perte pour la bibliothèque MTST."""
import torch
from torch import nn
import torch.nn.functional as F


def _masked_mean(values, weights=None):
    # Moyenne pondérée, avec masque facultatif.
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp(min=1)


class R2BoostedLoss2(nn.Module):
    """MSE des niveaux + MSE des différences - corrélation temporelle.

    Accepte des tenseurs (batch, temps, ...), quelle que soit la forme
    des dimensions non temporelles. time_dim indique l'axe du temps.
    """

    def __init__(self, alpha=1.0, beta=0.8, gamma=0.2,
                 eps=1e-8, time_dim=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.time_dim = time_dim

    def forward(self, y_pred, y_true, mask=None):
        if y_pred.shape != y_true.shape:
            raise ValueError("Prediction and target shapes must match")
        if y_pred.ndim < 2:
            raise ValueError("Expected at least two dimensions (batch, time)")

        axis = self.time_dim % y_pred.ndim
        if axis == 0:
            raise ValueError("time_dim cannot be the batch axis (0)")

        # Déplacer le temps en position 1, puis aplatir les autres axes.
        yp = y_pred.movedim(axis, 1).reshape(y_pred.shape[0], y_pred.shape[axis], -1)
        yt = y_true.movedim(axis, 1).reshape(y_true.shape[0], y_true.shape[axis], -1)

        weights = None
        if mask is not None:
            try:
                mask = torch.broadcast_to(mask, y_pred.shape)
            except RuntimeError as exc:
                raise ValueError("Mask cannot broadcast to prediction shape") from exc
            weights = (mask > 0).to(dtype=yp.dtype).movedim(axis, 1).reshape_as(yp)

        level = _masked_mean((yp - yt).square(), weights)
        zero = yp.sum() * 0.0
        if yp.shape[1] < 2:
            return self.alpha * level + self.beta * zero - self.gamma * zero

        # Erreur sur les variations entre pas temporels consécutifs.
        delta = (yp[:, 1:] - yp[:, :-1]) - (yt[:, 1:] - yt[:, :-1])
        adjacent = None if weights is None else weights[:, 1:] * weights[:, :-1]
        diff = _masked_mean(delta.square(), adjacent)

        # Corrélation de Pearson : une valeur par (batch, variable).
        w = torch.ones_like(yp) if weights is None else weights
        counts = w.sum(dim=1, keepdim=True)
        means_p = (yp * w).sum(dim=1, keepdim=True) / counts.clamp(min=1)
        means_t = (yt * w).sum(dim=1, keepdim=True) / counts.clamp(min=1)
        cp = (yp - means_p) * w
        ct = (yt - means_t) * w
        num = (cp * ct).sum(dim=1)
        denom = cp.square().sum(dim=1).sqrt() * ct.square().sum(dim=1).sqrt()
        corr_each = num / denom.clamp(min=self.eps)
        valid = (counts.squeeze(1) >= 2) & (denom > self.eps)
        corr = (corr_each * valid.to(corr_each.dtype)).sum() / valid.sum().clamp(min=1)

        return self.alpha * level + self.beta * diff - self.gamma * corr


class RegressionLoss(nn.Module):
    """Loss interface with an optional validity mask."""
    def __init__(self, name='mse', **kwargs):
        super().__init__()
        self.name = name.lower()
        if self.name not in {'mse', 'mae', 'huber', 'r2boosted'}:
            raise ValueError(f'unknown loss {name!r}; choose mse, mae, huber, r2boosted')
        self.custom = R2BoostedLoss2(**kwargs) if self.name == 'r2boosted' else None
        self.delta = kwargs.get('delta', 1.0) if self.name == 'huber' else None
        if self.name == 'huber' and set(kwargs) - {'delta'}:
            raise ValueError('huber accepts only delta')
        if self.name in {'mae','mse'} and kwargs:
            raise ValueError(f'{self.name} does not accept extra arguments')

    def forward(self, y_pred, y_true, mask=None):
        if self.custom is not None:
            return self.custom(y_pred, y_true, mask)
        if y_pred.shape != y_true.shape:
            raise ValueError('Prediction and target shapes must match')
        if self.name == 'mse':
            per_element = (y_pred - y_true).square()
        elif self.name == 'mae':
            per_element = (y_pred - y_true).abs()
        else:
            per_element = F.huber_loss(y_pred, y_true, reduction='none', delta=self.delta)
        if mask is not None:
            try:
                mask = torch.broadcast_to(mask, per_element.shape)
            except RuntimeError as exc:
                raise ValueError('mask cannot broadcast to prediction shape') from exc
            mask = (mask > 0).to(per_element.dtype)
        return _masked_mean(per_element, mask)


def get_loss(name='mse', **kwargs):
    """Example: get_loss('r2boosted', alpha=1., beta=.8, gamma=.2)."""
    return RegressionLoss(name, **kwargs)
