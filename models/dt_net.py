import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.GraphModels import GraphNeuralNetwork


class ModalityNormalization(nn.Module):
    """
    batch*frame*features
    """

    def __init__(self):
        super(ModalityNormalization, self).__init__()

    def forward(self, inputBatch):
        meanBatch = torch.mean(inputBatch, dim=2, keepdim=True)
        varBatch = torch.std(inputBatch, dim=2, keepdim=True)
        return (inputBatch - meanBatch) / varBatch


class DTNet(nn.Module):
    """
    """

    def __init__(self, dModel, graph_layer, druginSize, mlp_depth, graph_depth, GAT_head, targetinSize, pretrain_dir):
        super(DTNet, self).__init__()

        # drug-GNN
        self.drug_net = GraphNeuralNetwork(
            in_dim=druginSize,
            out_dim=dModel,
            layer_type=graph_layer,
            num_pre=mlp_depth,
            num_graph_layer=graph_depth,
            head=GAT_head
        )
        self.drug_conv = nn.Sequential(
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        )
        # target-pretrained model
        self.target_net = None
        self.target_conv = nn.Sequential(
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU(),
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        )

        # fusion
        self.outputConv = nn.Conv1d(dModel, 2, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, druginputBatch, targetinputBatch):
        druginputBatch = self.drug_net(druginputBatch[0], druginputBatch[1])
        targetinputBatch = self.target_net(targetinputBatch)

        jointBatch = None
        jointBatch = self.outputConv(jointBatch)
        return jointBatch
