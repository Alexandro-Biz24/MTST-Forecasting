import pytest
import torch
from mtst import MTSTForecaster, get_loss

@pytest.mark.parametrize('shape', [(24,), (24, 3), (24, 3, 2)])
def test_loss_backward_shapes(shape):
    pred = torch.randn(3, *shape, requires_grad=True)
    truth = torch.randn_like(pred)
    mask = torch.ones_like(pred)
    mask[:, 0] = 0
    value = get_loss('r2boosted', time_dim=1)(pred, truth, mask)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(pred.grad).all()


def test_time_axis_configurable():
    pred = torch.randn(3, 4, 12, requires_grad=True)
    truth = torch.randn_like(pred)
    value = get_loss('r2boosted', time_dim=2)(pred, truth)
    value.backward()
    assert torch.isfinite(pred.grad).all()


def test_fit_multidimensional_custom_loss():
    model = MTSTForecaster(input_dim=3, output_shape=(4, 2), embed_dim=16,
                           heads=4, n_layers=1, resolutions=(1, 2, 4))
    X, y = torch.randn(4, 8, 3), torch.randn(4, 4, 2)
    losses = model.fit(X, y, loss='r2boosted', loss_kwargs={'time_dim': 1},
                       epochs=1, batch_size=2, device='cpu', verbose=False)
    assert len(losses) == 1 and torch.isfinite(torch.tensor(losses)).all()
