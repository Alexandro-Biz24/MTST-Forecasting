import torch
import pytest
from mtst import MTSTForecaster

@pytest.mark.parametrize('output_shape', [(4,), (4, 2), (4, 2, 2)])
def test_fit_predict_save_load(tmp_path, output_shape):
    model = MTSTForecaster(input_dim=3, output_shape=output_shape, embed_dim=16,
                            heads=4, n_layers=1, resolutions=(1, 2, 4))
    X = torch.randn(5, 8, 3)
    y = torch.randn(5, *output_shape)
    mask = torch.ones_like(y)
    mask[0] = 0
    old = model(X).detach().clone()
    history = model.fit(X, y, mask=mask, epochs=1, batch_size=2,
                        lr=0.001, device='cpu', verbose=False)
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history)).all()
    new = model.predict(X, batch_size=2, device='cpu')
    assert new.shape == y.shape
    assert not torch.equal(old, new)
    filepath = tmp_path / 'model.pt'
    model.save(filepath)
    reloaded = MTSTForecaster.load(filepath, device='cpu')
    assert torch.allclose(new, reloaded.predict(X, device='cpu'), atol=1e-5)


def test_custom_loss_fit():
    m = MTSTForecaster(input_dim=2, output_shape=(4,), embed_dim=16, heads=4, n_layers=1)
    X, y = torch.randn(3, 8, 2), torch.randn(3, 4)
    assert len(m.fit(X, y, epochs=1, batch_size=2, loss='r2boosted', device='cpu', verbose=False)) == 1
