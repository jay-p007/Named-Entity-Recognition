import torch
import torch.nn as nn

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    def forward(self, logits, labels):
        loss_fn = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-100)
        return loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights
    def forward(self, logits, labels):
        ce_loss = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-100, reduction='none')(
            logits.view(-1, logits.shape[-1]), labels.view(-1)
        )
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()