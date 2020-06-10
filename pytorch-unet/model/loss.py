import torch.nn.functional as F


def cel_loss(output, target):
    return F.cross_entropy(output, target)
