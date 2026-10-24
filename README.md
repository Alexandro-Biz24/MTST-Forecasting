# MTST Forecasting

A configurable **Multi-Resolution Time Series Transformer (MTST)** implemented in PyTorch for forecasting. This is an independent implementation inspired by research on multi-resolution time-series Transformers, including work presented at AISTATS 2024. See the [Citation and attribution](#citation-and-attribution) section before publishing a research claim.

The package provides three parallel temporal-resolution branches, configurable forecasting output shapes, relative positional attention, and a simple training/inference API.

## Installation

After publication to PyPI:

```bash
pip install mtst-forecasting
```

For local development from this repository:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import torch
from mtst import MTSTForecaster

# User-prepared windows: samples x timesteps x features
X = torch.randn(100, 24, 8)
y = torch.randn(100, 24)

model = MTSTForecaster(
    input_dim=8,
    output_shape=(24,),
    embed_dim=64,
    heads=8,
    n_layers=2,
    dropout=0.1,
    resolutions=(1, 2, 4),
    positional_encoding="relative",
)

history = model.fit(
    X, y,
    epochs=10,
    batch_size=16,
    lr=5e-4,
    loss="mse",
    device="auto",
)

pred = model.predict(X[:5])  # torch.Tensor, shape (5, 24), on CPU

model.save("mtst_model.pt")
restored = MTSTForecaster.load("mtst_model.pt")
```

`device="auto"` chooses CUDA, Apple MPS, or CPU depending on availability.

## Shapes and multi-resolution settings

- Input: `(batch, sequence_length, input_dim)`.
- Output: `(batch, *output_shape)`; supported examples are `(24,)`, `(24, 3)` and `(24, 3, 2)`.
- `resolutions=(1, 2, 4)` selects every 1st, 2nd and 4th timestamp for separate encoder branches. These are **sampling strides**, not trainable token aggregators.
- `positional_encoding="relative"` or `"both"` (relative attention plus absolute position encoding).

## Training losses

Available choices: `"mse"`, `"mae"`, `"huber"`, and `"r2boosted"`.

```python
history = model.fit(
    X, y,
    loss="r2boosted",
    loss_kwargs={"alpha": 1.0, "beta": 0.8, "gamma": 0.2, "time_dim": 1},
    epochs=10,
)
```

`r2boosted` combines a level error term, differences along the chosen temporal axis, and a correlation-based term. `time_dim=1` by default, where axis 0 is the batch dimension. You can optionally supply `mask` with the same shape as `y` to exclude invalid positions. Standard losses do not need `time_dim`.

## Responsibilities and limitations

The library **does not** create features, perform scaling, select time windows, or split datasets. Prepare your training targets and inputs yourself, avoid look-ahead leakage, and save any preprocessing scalers separately. `fit` trains the model in mini-batches; `predict` is inference-only. The standard PyTorch interface `model(X)` is also available.

Saved checkpoints contain model configuration and learned weights, not the optimizer state or preprocessing transforms. Only load checkpoints from trusted sources.

## Development and tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
```

## Citation and attribution

This library is an independent PyTorch implementation inspired by multi-resolution forecasting research. Add the exact bibliographic reference to the relevant AISTATS 2024 paper and clearly document differences before publication. The repository is not claimed to be the authors' official implementation.

## Package Information

MTST Forecasting is an open-source PyTorch library for configurable multi-resolution time series forecasting.

You can retrieve the installed package version and author directly in Python:

```python
import mtst

print(mtst.__version__)
print(mtst.__author__)
```

Additional package metadata can be accessed using:

```python
from importlib.metadata import metadata

info = metadata("mtst-forecasting")

print("Package:", info["Name"])
print("Version:", info["Version"])
print("Author:", info["Author"])
```

For source code, documentation, and contributions, visit the GitHub repository linked in the package metadata.

## License

MIT — see [LICENSE](LICENSE).
