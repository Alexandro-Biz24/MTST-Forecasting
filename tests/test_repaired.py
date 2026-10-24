import torch
import pytest
from mtst import MTSTForecaster
from mtst.Models.transformer_basis import TransformerBlock, CustomTransformerEncoder
from mtst.training import _rte_load_mape

@pytest.mark.parametrize('encoding,klass', [('absolute',TransformerBlock),('relative',CustomTransformerEncoder),('both',CustomTransformerEncoder)])
def test_positional_encodings(encoding, klass):
    model = MTSTForecaster(input_dim=3,output_shape=(6,),embed_dim=16,heads=4,n_layers=1,resolutions=(1,2,4),positional_encoding=encoding)
    assert isinstance(model.backbone.transformer_high_res, klass)
    x = torch.randn(2,6,3)
    pred = model(x)
    assert pred.shape == (2,6)
    assert torch.isfinite(pred).all()
    pred.mean().backward()
    assert model.output_head.weight.grad is not None

@pytest.mark.parametrize('encoding', ['absolute','relative'])
def test_fit_mask_and_logs(encoding, capsys):
    torch.manual_seed(23)
    model = MTSTForecaster(input_dim=3, output_shape=(6,), embed_dim=16, heads=4,n_layers=1,positional_encoding=encoding)
    X=torch.randn(5,6,3);y=torch.randn(5,6);mask=torch.ones_like(y);mask[:,0]=0
    history=model.fit(X,y,mask=mask,epochs=1,batch_size=2,device='cpu',log_every=1)
    assert len(history)==1 and torch.isfinite(torch.tensor(history)).all()
    out=capsys.readouterr().out
    assert 'Batch 1/' in out and 'Training completed' in out

def test_rte_mape_matches_original_formula():
    x=torch.tensor([[[0.,1.],[1.,0.]]]);y=torch.tensor([[1.,0.]])
    pred=torch.tensor([[2.,1.]],requires_grad=True)
    mask=torch.tensor([[1.,0.]])
    opts={'feature_index':1,'x_mean':100.,'x_scale':10.,'y_mean':5.,'y_scale':2.}
    loss=_rte_load_mape(pred,y,mask,x,opts)
    rte=x[:,:,1]*10+100
    true=rte+y*2+5
    expected=((((rte+pred*2+5)-true).abs()/true.abs().clamp(min=1))*mask).sum()/mask.sum()
    assert torch.allclose(loss,expected)
    loss.backward()
    assert pred.grad[0,0].abs()>0

def test_fit_rte_mape():
    model=MTSTForecaster(input_dim=3,output_shape=(4,),embed_dim=16,heads=4,n_layers=1,positional_encoding='absolute')
    X=torch.randn(4,4,3);y=torch.randn(4,4);mask=torch.ones_like(y)
    h=model.fit(X,y,mask=mask,epochs=1,batch_size=2,device='cpu',loss='rte_mape',loss_kwargs={'feature_index':0,'x_mean':1000.,'x_scale':100.,'y_mean':0.,'y_scale':50.},verbose=False)
    assert len(h)==1 and h[0]>=0

def test_save_load(tmp_path):
    model=MTSTForecaster(input_dim=3,output_shape=(4,),embed_dim=16,heads=4,n_layers=1,positional_encoding='absolute')
    path=tmp_path/'model.pt';model.save(path)
    loaded=MTSTForecaster.load(path,device='cpu')
    assert loaded.config['positional_encoding']=='absolute'
    assert loaded(torch.zeros(1,4,3)).shape==(1,4)
