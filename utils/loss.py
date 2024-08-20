import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        """
        :param gamma: focusing parameter (default=2)
        :param alpha: balancing factor, set alpha to None for equal importance to all classes
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert the class indices into one-hot vectors
        targets = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        # Compute the cross entropy loss
        logpt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(logpt)  # pt is the predicted probability
        logpt = logpt * targets
        logpt = logpt.sum(dim=-1)

        # Compute the focal loss
        loss = -1 * (1 - pt) ** self.gamma * logpt

        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            loss *= self.alpha[targets.argmax(dim=-1)]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
