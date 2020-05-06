# -*- coding: utf-8 -*-
import torch
import numpy as np

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
b = a.numpy()
print(b)

# memory shared
a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
