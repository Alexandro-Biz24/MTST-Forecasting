import math
import torch
from torch import nn
from mtst.Models.full_model import MTST_full
from mtst.training import fit_model, predict_model, save_model, load_model


class MTSTForecaster(nn.Module):
    """Configurable multi-resolution forecaster.

    Input: [batch, sequence_length, input_dim].
    Output: [batch, *output_shape].
    """

    def __init__(self, input_dim, output_shape=(24,), embed_dim=64,
                 heads=8, n_layers=2, dropout=0.1,
                 resolutions=(1, 2, 4), max_len=5000, positional_encoding="relative"):
        super().__init__()
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        if not output_shape or any(not isinstance(d, int) or d <= 0 for d in output_shape):
            raise ValueError('output_shape must contain positive integers')
        if len(resolutions) != 3 or any(not isinstance(r, int) or r < 1 for r in resolutions):
            raise ValueError('resolutions must contain three positive integers')
        if input_dim < 1 or embed_dim < 1 or heads < 1 or embed_dim % heads != 0:
            raise ValueError('invalid input_dim, embed_dim, or heads')
        self.config = dict(input_dim=input_dim, output_shape=tuple(output_shape),
                           embed_dim=embed_dim, heads=heads, n_layers=n_layers,
                           dropout=dropout, resolutions=tuple(resolutions),
                           max_len=max_len, positional_encoding=positional_encoding)
        self.input_dim = input_dim
        self.output_shape = tuple(output_shape)
        self.resolutions = tuple(resolutions)
        self.backbone = MTST_full(
            input_dim=input_dim, embed_dim=embed_dim, heads=heads,
            n_layers=n_layers, dropout=dropout, output_len=output_shape[0],
            max_len=max_len, positional_encoding=positional_encoding,
        )
        self.output_head = nn.Linear(input_dim, math.prod(output_shape[1:]))

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f'expected [batch, time, {self.input_dim}]')
        if x.shape[1] < 1:
            raise ValueError('sequence_length must be >= 1')
        x = self.backbone(x, *self.resolutions)
        x = self.output_head(x)
        return x.reshape(x.shape[0], *self.output_shape)

    def fit(self, X, y, **kwargs):
        # Entraînement complet : batches, gradients et optimiseur.
        self.history_ = fit_model(self, X, y, **kwargs)
        return self.history_

    def predict(self, X, **kwargs):
        # Inférence sans gradients ; sortie sur CPU.
        return predict_model(self, X, **kwargs)

    def save(self, path):
        # Sauvegarde l'architecture et ses poids.
        save_model(self, path)

    @classmethod
    def load(cls, path, **kwargs):
        # Reconstruit le modèle à partir du checkpoint.
        return load_model(cls, path, **kwargs)
