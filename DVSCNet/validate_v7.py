import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_DVSCNet import DVSCNet, compute_loss


torch.manual_seed(17)
model = DVSCNet(4, auxiliary_dim=3)

inputs = torch.randn(2, 1, 64, 157)
auxiliary_features = torch.randn(2, 3)
labels = torch.tensor([0, 1])
outputs = model(inputs, auxiliary_features)
loss = compute_loss(outputs[0], labels) + model.regularization_loss(outputs)
loss.backward()

assert outputs[0].shape == (2, 4)
assert all(tensor.shape == (2, 32) for tensor in outputs[1:])
assert all(torch.isfinite(tensor).all() for tensor in outputs)
auxiliary_gradients = [
	parameter.grad
	for parameter in model.auxiliary_encoder.parameters()
	if parameter.requires_grad
]
assert all(gradient is not None for gradient in auxiliary_gradients)
assert all(torch.isfinite(gradient).all() for gradient in auxiliary_gradients)
assert sum(gradient.abs().sum().item() for gradient in auxiliary_gradients) > 0

print("DVSCNet auxiliary feature fusion gradient OK")
print(f"params={sum(parameter.numel() for parameter in model.parameters())}")