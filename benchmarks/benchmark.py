"""Standalone forward / inference / backward+optimizer / fit benchmark.
Run: PYTHONPATH=src python benchmarks/benchmark.py
"""
import time
import torch
from mtst import MTSTForecaster
from mtst.training import resolve_device

D = resolve_device('auto')
torch.manual_seed(42)
model = MTSTForecaster(input_dim=8, output_shape=(24,),embed_dim=64,heads=4,n_layers=2,dropout=0.1,positional_encoding='absolute').to(D)
x=torch.randn(8,24,8,device=D)
y=torch.randn(8,24,device=D)
opt=torch.optim.Adam(model.parameters(),lr=5e-4)

def sync():
    if D.type=='mps': torch.mps.synchronize()
    elif D.type=='cuda': torch.cuda.synchronize()

def measure(label, func, repeat=10):
    for _ in range(2):func()
    sync();start=time.perf_counter()
    for _ in range(repeat):func()
    sync();print(f'{label}: {(time.perf_counter()-start)*1000/repeat:.2f} ms/batch',flush=True)

def forward():
    with torch.no_grad(): model(x)

def inference():
    with torch.inference_mode():model(x)

def train_step():
    opt.zero_grad(set_to_none=True)
    pred=model(x)
    loss=torch.nn.functional.mse_loss(pred,y)
    loss.backward()
    opt.step()

print(f'Device: {D}',flush=True)
model.eval(); measure('Forward no_grad',forward); measure('Inference',inference)
model.train();measure('Forward + backward + optimizer',train_step)
print('One-epoch API fit (100 days):',flush=True)
model.fit(torch.randn(100,24,8),torch.randn(100,24),epochs=1,batch_size=8,device=str(D),log_every=5)
