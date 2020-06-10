import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
        eleSize = 1
        for sz in target.shape:
            eleSize = eleSize * sz
    return correct / eleSize
