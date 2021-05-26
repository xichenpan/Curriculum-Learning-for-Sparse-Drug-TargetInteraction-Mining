from models.GraphModels import GraphNeuralNetwork
from models.attn import *
from utils.protein_embedding import *
from .convlist import ConvFeatureExtractionModel


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

    def __init__(self, freeze_protein_embedding, dModel, graph_layer, druginSize, mlp_depth, graph_depth, GAT_head, targetinSize, pretrain_dir,
                 device, model_name, drug_conv, target_conv, conv_dropout):
        super(DTNet, self).__init__()
        self.freeze_protein_embedding = freeze_protein_embedding
        self.model_name = model_name
        # drug-GNN
        self.drug_net = GraphNeuralNetwork(
            in_dim=druginSize,
            out_dim=dModel,
            layer_type=graph_layer,
            num_pre=mlp_depth,
            num_graph_layer=graph_depth,
            head=GAT_head
        )

        self.drug_conv_list = drug_conv
        drug_conv = eval(drug_conv)
        self.drugConv = ConvFeatureExtractionModel(dModel, drug_conv, conv_dropout)

        # target-pretrained model
        if not freeze_protein_embedding:
            lm = BiLM(nin=22, embedding_dim=21, hidden_dim=1024, num_layers=2, nout=21)
            model_ = StackedRNN(nin=21, nembed=512, nunits=512, nout=100, nlayers=3, padding_idx=20, dropout=0, lm=lm)
            model = OrdinalRegression(embedding=model_, n_classes=5)
            state = torch.load(pretrain_dir)
            model.load_state_dict(state)
            self.target_net = load_model(model, device=device)

        self.target_conv_list = target_conv
        target_conv = eval(target_conv)
        self.targetConv = ConvFeatureExtractionModel(targetinSize, target_conv, conv_dropout)

        # cross attention
        if model_name == "cross_attn":
            self.cross_attn_module = Drug_Target_Cross_Attnention_Pooling(drug_feature_dim=512, target_feature_dim=512, layer_num=None,
                                                                          proj_bias=True)
        # fusion
        self.DrugModalityNormalization = ModalityNormalization()
        self.TargetModalityNormalization = ModalityNormalization()

        self.outputMLP = nn.Sequential(
            nn.Linear(dModel * 2, dModel),
            nn.ReLU(),
            nn.Linear(dModel, 2)
        )
        return

    def _get_feat_extract_output_lengths(self, convlist, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(convlist)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2])

        return input_lengths.to(torch.long)

    def forward(self, druginputBatch, targetinputBatch):
        drug_padding_mask = druginputBatch[2]
        druginputBatch = self.drug_net(druginputBatch[0], druginputBatch[1])
        druginputBatch = druginputBatch.transpose(1, 2)
        druginputBatch = self.drugConv(druginputBatch)
        druginputBatch = druginputBatch.transpose(2, 1)
        drug_len = torch.sum(~drug_padding_mask, dim=1)
        drug_len = self._get_feat_extract_output_lengths(self.drug_conv_list, drug_len.long())
        drug_len = drug_len.unsqueeze(-1).expand(list(drug_len.shape) + [druginputBatch.shape[-1]])
        drug_padding_mask = drug_padding_mask.unsqueeze(-1).expand(list(drug_padding_mask.shape) + [druginputBatch.shape[-1]])
        druginputBatch = druginputBatch * ~drug_padding_mask

        target_padding_mask = targetinputBatch[1]
        targetinputBatch = targetinputBatch[0]
        if not self.freeze_protein_embedding:
            with torch.no_grad():
                targetinputBatch = embedding(targetinputBatch.to(torch.int64), self.target_net, targetinputBatch.device)
        else:
            targetinputBatch = targetinputBatch.float()
        targetinputBatch = targetinputBatch.transpose(1, 2)
        targetinputBatch = self.targetConv(targetinputBatch)
        targetinputBatch = targetinputBatch.transpose(2, 1)
        target_len = torch.sum(~target_padding_mask, dim=1)
        target_len = self._get_feat_extract_output_lengths(self.target_conv_list, target_len.long())
        target_len = target_len.unsqueeze(-1).expand(list(target_len.shape) + [targetinputBatch.shape[-1]])
        target_padding_mask = target_padding_mask.unsqueeze(-1).expand(list(target_padding_mask.shape) + [targetinputBatch.shape[-1]])
        targetinputBatch = targetinputBatch * ~target_padding_mask

        if self.model_name == "baseline":
            # bz * token *dim
            druginputBatch = druginputBatch.sum(1) / drug_len
            targetinputBatch = targetinputBatch.sum(1) / target_len
            druginputBatch = self.DrugModalityNormalization(druginputBatch)
            targetinputBatch = self.TargetModalityNormalization(targetinputBatch)
        else:
            druginputBatch, targetinputBatch = self.cross_attn_module(druginputBatch, targetinputBatch, drug_padding_mask, target_padding_mask)
            druginputBatch = druginputBatch.squeeze(1)  # bs * 512
            targetinputBatch = targetinputBatch.squeeze(1)  # bs * 512

        jointBatch = torch.cat([druginputBatch, targetinputBatch], dim=1)
        jointBatch = self.outputMLP(jointBatch)

        return jointBatch
