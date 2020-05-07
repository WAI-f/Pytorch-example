# -*- coding: utf-8 -*-
import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# Do a tensor operation
y = x + 2
print(y)

# change requires_grad flag in-place
a = torch.randn(2, 2)
print(a.grad_fn)
a = ((a * 3) / (a - 1))
print(a.grad_fn)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
a = ((a * 3) / (a - 1))
print(a.grad_fn)
