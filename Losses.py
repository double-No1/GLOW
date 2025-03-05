import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OCC_loss(nn.Module):
    def __init__(self, ord=2,):
        super().__init__()
        self.ord = ord
    def forward(self, z_c, center, eval=False):

        # z = z_c[0]-z_c[1]
        # diffs = torch.pow(z.norm(p=2,dim=-1),self.ord)
        diffs = torch.pow((z_c - center).norm(p=self.ord, dim=-1), self.ord)
        if eval:

            return diffs
        else:
            return diffs



# 定义 InfoNCE 损失函数
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features1, features2):
        """
        Computes the InfoNCE loss.
        Args:
            features1: output of the first model (batch_size, feature_dim)
            features2: output of the second model (batch_size, feature_dim)
        Returns:
            loss: the InfoNCE loss value
        """
        batch_size = features1.size(0)


        # Cosine similarity matrix
        similarity_matrix = torch.mm(features1, features2.T) / self.temperature

        # Mask out same-sample pairs
        mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        positives = similarity_matrix[mask].view(batch_size, 1)
        negatives = similarity_matrix[~mask].view(batch_size, -1)

        # Compute InfoNCE loss
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=similarity_matrix.device)
        loss = F.cross_entropy(logits, labels)

        return loss