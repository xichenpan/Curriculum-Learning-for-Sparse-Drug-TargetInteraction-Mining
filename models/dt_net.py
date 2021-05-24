import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.GraphModels import GraphNeuralNetwork
from utils.protein_embedding import *
from models.attn import *
import time


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

    def __init__(self, dModel, graph_layer, druginSize, mlp_depth, graph_depth, GAT_head, targetinSize, pretrain_dir,
                 device, model_name="cross_attn"):
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
        self.drugConv = nn.Conv1d(dModel, dModel, kernel_size=(1,), stride=(1,), padding=(0,))
        self.drugPost = nn.Sequential(
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
        lm = BiLM(nin=22, embedding_dim=21, hidden_dim=1024, num_layers=2, nout=21)
        model_ = StackedRNN(nin=21, nembed=512, nunits=512, nout=100, nlayers=3, padding_idx=20, dropout=0, lm=lm)
        model = OrdinalRegression(embedding=model_, n_classes=5)
        state = torch.load(pretrain_dir)
        model.load_state_dict(state)
        self.target_net = load_model(model, device=device)
        self.targetConv = nn.Conv1d(targetinSize, dModel, kernel_size=(1,), stride=(1,), padding=(0,))
        self.targetPost = nn.Sequential(
            nn.LayerNorm(dModel),
            nn.ReLU()
        )

        self.model_name = model_name

        # cross attention
        if model_name == "cross_attn":
            self.cross_attn_module = Drug_Target_Cross_Attnention_Pooling(
                drug_feature_dim=512,
                target_feature_dim=512,
                layer_num=None,
                proj_bias=True)

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
        self.DrugModalityNormalization = ModalityNormalization()
        self.TargetModalityNormalization = ModalityNormalization()

        self.outputMLP = nn.Sequential(
            nn.Linear(dModel * 2, dModel),
            nn.ReLU(),
            nn.Linear(dModel, 2)
        )
        return

    def forward(self, druginputBatch, targetinputBatch):
        # targetinputBatch = embedding(targetinputBatch, self.pretrained_model, device)

        drug_padding_mask = druginputBatch[2]
        druginputBatch = self.drug_net(druginputBatch[0], druginputBatch[1])
        druginputBatch = druginputBatch.transpose(1, 2)
        druginputBatch = self.drugConv(druginputBatch)
        druginputBatch = druginputBatch.transpose(2, 1)
        druginputBatch = self.drugPost(druginputBatch)
        drug_len = torch.sum(~drug_padding_mask, dim=1)
        drug_len = drug_len.unsqueeze(-1).expand(list(drug_len.shape) + [druginputBatch.shape[-1]])

        target_padding_mask = targetinputBatch[1]
        targetinputBatch = targetinputBatch[0]

        # time_embdedding = time.time()
        with torch.no_grad():
            targetinputBatch = embedding(targetinputBatch.to(torch.int64), self.target_net, targetinputBatch.device)
            # print("TIME EMBD: ", time.time() - time_embdedding)
        targetinputBatch = targetinputBatch.transpose(1, 2)
        targetinputBatch = self.targetConv(targetinputBatch)

        targetinputBatch = targetinputBatch.transpose(2, 1)
        targetinputBatch = self.targetPost(targetinputBatch)
        target_len = torch.sum(~target_padding_mask, dim=1)
        target_len = target_len.unsqueeze(-1).expand(list(target_len.shape) + [targetinputBatch.shape[-1]])

        drug_padding_mask = drug_padding_mask.unsqueeze(-1).expand(
            list(drug_padding_mask.shape) + [druginputBatch.shape[-1]])
        target_padding_mask = target_padding_mask.unsqueeze(-1).expand(
            list(target_padding_mask.shape) + [targetinputBatch.shape[-1]])
        druginputBatch = druginputBatch * ~drug_padding_mask
        targetinputBatch = targetinputBatch * ~target_padding_mask

        if self.model_name == "baseline":
            # bz * token *dim
            druginputBatch = druginputBatch.sum(1) / drug_len
            targetinputBatch = targetinputBatch.sum(1) / target_len
            druginputBatch = self.DrugModalityNormalization(druginputBatch)
            targetinputBatch = self.TargetModalityNormalization(targetinputBatch)
        else:
            druginputBatch, targetinputBatch = self.cross_attn_module(
                druginputBatch,
                targetinputBatch,
                drug_padding_mask,
                target_padding_mask
            )
            # print(druginputBatch.shape, targetinputBatch.shape)
            druginputBatch = druginputBatch.squeeze(1)  # bs * 512
            targetinputBatch = targetinputBatch.squeeze(1)  # bs * 512

        jointBatch = torch.cat([druginputBatch, targetinputBatch], dim=1)
        jointBatch = self.outputMLP(jointBatch)

        return jointBatch
