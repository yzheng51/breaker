import torch


def kl_divergence_loss(y_pred, y_true):
    return torch.mean(torch.sum(y_true * torch.log(y_true / y_pred), axis=1))
