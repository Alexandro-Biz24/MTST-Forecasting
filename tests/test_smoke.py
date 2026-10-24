import torch
from mtst import MTSTForecaster, get_loss


def test_forward_backward():
    for output_shape in [(24,), (24, 3), (24, 3, 2)]:
        model = MTSTForecaster(input_dim=8, output_shape=output_shape,
                               embed_dim=32, heads=4, n_layers=1, resolutions=(1, 2, 4))
        x = torch.randn(2, 24, 8)
        y = torch.randn(2, *output_shape)
        pred = model(x)
        assert pred.shape == y.shape
        for loss_name in ['mse', 'mae', 'huber']:
            model.zero_grad()
            loss = get_loss(loss_name)(model(x), y)
            loss.backward()
            assert torch.isfinite(loss)
        if len(output_shape) == 1:
            model.zero_grad()
            mask = torch.ones_like(y)
            mask[:, 2] = 0
            loss = get_loss('r2boosted')(model(x), y, mask)
            loss.backward()
            assert torch.isfinite(loss)


def test_encoding_modes():
    for mode in ("relative", "both"):
        model = MTSTForecaster(input_dim=3, output_shape=(4,), embed_dim=16, heads=4, n_layers=1, positional_encoding=mode)
        assert model(torch.randn(2, 12, 3)).shape == (2, 4)
