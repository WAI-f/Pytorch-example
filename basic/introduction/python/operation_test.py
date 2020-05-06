# -*- coding: utf-8 -*-
import torch

# Addition:syntax 1
x = torch.ones(5, 3)
y = torch.rand(5, 3)
print(x + y)

# Addition: syntax 2
x = torch.ones(5, 3)
y = torch.rand(5, 3)
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addition: syntax 3, in-place
x = torch.ones(5, 3)
y = torch.rand(5, 3)
y.add_(x)
print(y)

# Resizing: If you want to resize/reshape tensor:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size())
print(y.size())
print(z.size())

# one element tensor, use .item() to get the value
x = torch.randn(1)
print(x)
print(x.item())
