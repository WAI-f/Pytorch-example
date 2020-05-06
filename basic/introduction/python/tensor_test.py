# -*- coding: utf-8 -*-
import torch

# Construct a 5x3 matrix, uninitialized:
x = torch.empty(5, 3)
print(x)

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

# Construct a matrix filled zeros and of dtype long:
x = torch.zeros((5, 3), dtype=torch.long)
print(x)

# Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x)
print(x.size())
