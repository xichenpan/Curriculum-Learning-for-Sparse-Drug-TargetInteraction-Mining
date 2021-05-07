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
        self.graph_model = GraphNeuralNetwork(
            in_dim=druginSize,
            out_dim=dModel,
            layer_type=graph_layer,
            num_pre=mlp_depth,
            num_graph_layer=graph_depth,
            head=GAT_head
        )
        # target-pretrained model

        # fusion
        self.outputConv = nn.Conv1d(dModel, 2, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, druginputBatch, targetinputBatch):
        druginputBatch = self.graph_model(druginputBatch[0], druginputBatch[1])

        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        # print(F.softmax(jointBatch,2))
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return outputBatch
