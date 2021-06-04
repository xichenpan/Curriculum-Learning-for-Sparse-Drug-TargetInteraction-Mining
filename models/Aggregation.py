import torch
import torch.nn as nn


class WeighedSumAndMax(nn.Module):
    def __init__(self, dim):
        super(WeighedSumAndMax, self).__init__()

        self.weightlayer = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        self.out = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.GELU()
        )

    def forward(self, node, paddingMask):
        if len(paddingMask.shape) != len(node.shape):
            paddingMask = paddingMask.unsqueeze(-1)
        weight = self.weightlayer(node)
        weightedNode = weight * node
        f_weightedSum = torch.sum(weightedNode * ~paddingMask, 1)

        node = torch.masked_fill(node, paddingMask.bool(), -1000)
        f_max = torch.max(node, 1)[0]

        graph = torch.cat([f_max, f_weightedSum], 1)
        return self.out(graph)


