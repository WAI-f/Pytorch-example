# -*- coding: utf-8 -*-
import torch

# network output a scalar
x = torch.randn(3, requires_grad=True)
y = x * 2
z = y.sum()
z.backward()
print(x.grad)

# network output a vector
x = torch.randn(3, requires_grad=True)
y = x * 2
v = torch.ones(3, dtype=torch.float)
y.backward(v)
print(x.grad)
