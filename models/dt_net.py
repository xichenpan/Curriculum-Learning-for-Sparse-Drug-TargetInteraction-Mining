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
        meanBatch = torch.mean(inputBatch, dim=1, keepdim=True)
        varBatch = torch.std(inputBatch, dim=1, keepdim=True)
        return (inputBatch - meanBatch) / varBatch


class DTNet(nn.Module):
    """
    """

    def __init__(self, dModel, graph_layer, druginSize, mlp_depth, graph_depth, GAT_head, targetinSize):
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
        self.drugConv = nn.Sequential(
            nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU()
        )
        # self.drug_conv = nn.Sequential(
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        # )
        # target-pretrained model
        self.target_net = None
        self.targetConv = nn.Sequential(
            nn.Conv1d(targetinSize, dModel, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(dModel),
            nn.ReLU()
        )
        # self.target_conv = nn.Sequential(
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        #     nn.LayerNorm(dModel),
        #     nn.ReLU(),
        #     nn.Conv1d(dModel, dModel, kernel_size=1, stride=1, padding=0),
        # )

        # fusion
        self.ModalityNormalization = ModalityNormalization()
        self.outputMLP = nn.Sequential(
            nn.Linear(dModel, dModel / 2),
            nn.ReLU(),
            nn.Linear(dModel, 2)
        )
        return

    def forward(self, druginputBatch, targetinputBatch):
        drug_padding_mask = ~druginputBatch[2]
        druginputBatch = self.drug_net(druginputBatch[0], druginputBatch[1])
        druginputBatch = druginputBatch.transpose(1, 2)
        druginputBatch = self.drugConv(druginputBatch)
        druginputBatch = druginputBatch.transpose(2, 1)
        drug_len = torch.sum(~drug_padding_mask, dim=1)
        drug_len = drug_len.unsqueeze(-1).expand(list(drug_len.shape).append(druginputBatch.shape[-1]))

        target_padding_mask = ~targetinputBatch[1]
        targetinputBatch = targetinputBatch[0]
        targetinputBatch = targetinputBatch.transpose(1, 2)
        targetinputBatch = self.drugConv(targetinputBatch)
        targetinputBatch = targetinputBatch.transpose(2, 1)
        target_len = torch.sum(~target_padding_mask, dim=1)
        target_len = target_len.unsqueeze(-1).expand(list(target_len.shape).append(targetinputBatch.shape[-1]))

        drug_padding_mask = drug_padding_mask.unsqueeze(-1).expand(
            list(drug_padding_mask.shape).append(druginputBatch.shape[-1]))
        target_padding_mask = target_padding_mask.unsqueeze(-1).expand(
            list(target_padding_mask.shape).append(targetinputBatch.shape[-1]))
        druginputBatch = druginputBatch*drug_padding_mask
        targetinputBatch = targetinputBatch*target_padding_mask

        druginputBatch = druginputBatch.mean(1)/drug_len
        targetinputBatch = targetinputBatch.mean(1)/target_len
        druginputBatch = self.ModalityNormalization(druginputBatch)
        targetinputBatch = self.ModalityNormalization(targetinputBatch)

        jointBatch = torch.cat([druginputBatch, targetinputBatch], dim=1)
        jointBatch = self.outputMLP(jointBatch)
        return jointBatch
