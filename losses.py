import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: (B, C, D, H, W) logits
        targets: (B, D, H, W) ground truth labels
        """
        preds = F.softmax(preds, dim=1)  # convert logits → probabilities
        targets_onehot = F.one_hot(targets, num_classes=preds.shape[1])  # (B, D, H, W, C)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        intersection = torch.sum(preds * targets_onehot, dim=(2, 3, 4))
        union = torch.sum(preds + targets_onehot, dim=(2, 3, 4))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # average over classes and batch
        

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.w_dice = weight_dice
        self.w_ce = weight_ce

    def forward(self, preds, targets):
        loss_dice = self.dice(preds, targets)
        loss_ce = self.ce(preds, targets)
        return self.w_dice * loss_dice + self.w_ce * loss_ce