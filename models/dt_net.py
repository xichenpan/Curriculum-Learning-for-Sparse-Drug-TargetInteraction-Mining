import math

from models.GraphModels import GraphNeuralNetwork
from models.attn import *
from utils.protein_embedding import *
from .convlist import ConvFeatureExtractionModel
from .Aggregation import WeighedSumAndMax


class PositionalEncoding(nn.Module):
    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float() * (math.log(10000.0) / dModel))
        pe[:, 0::2] = torch.sin(position / denominator)
        pe[:, 1::2] = torch.cos(position / denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0], :, :]
        return outputBatch


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
                 device, atten_type, drug_conv, target_conv, conv_dropout, add_transformer, focal_loss):
        super(DTNet, self).__init__()
        self.freeze_protein_embedding = freeze_protein_embedding
        self.atten_type = atten_type
        self.add_transformer = add_transformer
        self.focal_loss = focal_loss
        if focal_loss:
            final_dim = 1
        else:
            final_dim = 2
        if add_transformer:
            self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=1000)
            encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=4, dim_feedforward=1024, dropout=0.1)
            self.drugEncoder = nn.TransformerEncoder(encoderLayer, num_layers=4)
            self.targetEncoder = nn.TransformerEncoder(encoderLayer, num_layers=4)

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
        if atten_type == "cross_attn":
            self.cross_attn_module = Drug_Target_Cross_Attnention_Pooling(drug_feature_dim=512, target_feature_dim=512, layer_num=None,
                                                                          proj_bias=True)
        elif atten_type == "target2drug_attn":
            self.cross_attn_module = Target2Drug_Attnention_Block(drug_feature_dim=512, target_feature_dim=512, proj_bias=True)
        elif atten_type == 'wsam':
            self.drugWeightedSumAndMax = WeighedSumAndMax(dModel)
            self.targetWeightedSumAndMax = WeighedSumAndMax(dModel)
        
        # fusion
        # self.DrugModalityNormalization = ModalityNormalization()
        # self.TargetModalityNormalization = ModalityNormalization()

        self.outputMLP = nn.Sequential(
            nn.Linear(dModel * 2, dModel),
            nn.BatchNorm1d(dModel),
            nn.ReLU(),
            nn.Linear(dModel, dModel),
            nn.BatchNorm1d(dModel),
            nn.ReLU(),
            nn.Linear(dModel, final_dim)
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

    def forward(self, drugBatch, targetBatch):
        drug_padding_mask = drugBatch[2]
        drugBatch = self.drug_net(drugBatch[0], drugBatch[1])
        drugBatch = drugBatch.transpose(1, 2)
        drugBatch = self.drugConv(drugBatch)
        drugBatch = drugBatch.transpose(2, 1)
        # calc length
        drug_len = torch.sum(~drug_padding_mask, dim=1)
        # tx
        if self.add_transformer:
            drugBatch = drugBatch.transpose(1, 0)
            drugBatch = self.positionalEncoding(drugBatch)
            drugBatch = self.targetEncoder(drugBatch, src_key_padding_mask=drug_padding_mask)  # TBC
            drugBatch = drugBatch.transpose(0, 1)

        # unqueeeze and expand in feature dim
        drug_len = drug_len.unsqueeze(-1).expand(list(drug_len.shape) + [drugBatch.shape[-1]])
        drug_padding_mask = drug_padding_mask.unsqueeze(-1).expand(list(drug_padding_mask.shape) + [drugBatch.shape[-1]])
        drugBatch = drugBatch * ~drug_padding_mask

        target_padding_mask = targetBatch[1]
        targetBatch = targetBatch[0]
        if not self.freeze_protein_embedding:
            with torch.no_grad():
                targetBatch = embedding(targetBatch.to(torch.int64), self.target_net, targetBatch.device)
        else:
            targetBatch = targetBatch.float()
        targetBatch = targetBatch.transpose(1, 2)
        targetBatch = self.targetConv(targetBatch)  # BCT
        targetBatch = targetBatch.transpose(2, 1)
        # calc length
        target_len = torch.sum(~target_padding_mask, dim=1)
        target_len = self._get_feat_extract_output_lengths(self.target_conv_list, target_len.long())
        # rebuild target_padding_mask
        target_padding_mask = torch.zeros(targetBatch.shape[:-1], device=targetBatch.device)
        target_padding_mask[(torch.arange(targetBatch.shape[0]), target_len - 1)] = 1
        target_padding_mask = (1 - target_padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        # tx
        if self.add_transformer:
            targetBatch = targetBatch.transpose(1, 0)
            targetBatch = self.positionalEncoding(targetBatch)
            targetBatch = self.targetEncoder(targetBatch, src_key_padding_mask=target_padding_mask)  # TBC
            targetBatch = targetBatch.transpose(0, 1)
        # unqueeeze and expand in feature dim
        target_len = target_len.unsqueeze(-1).expand(list(target_len.shape) + [targetBatch.shape[-1]])
        target_padding_mask = target_padding_mask.unsqueeze(-1).expand(list(target_padding_mask.shape) + [targetBatch.shape[-1]])
        targetBatch = targetBatch * ~target_padding_mask

        if self.atten_type == "cross_attn":
            drugBatch, targetBatch = self.cross_attn_module(drugBatch, targetBatch, drug_padding_mask, target_padding_mask)
            drugBatch = drugBatch.squeeze(1)  # bs * 512
            targetBatch = targetBatch.squeeze(1)  # bs * 512
        elif self.atten_type == "target2drug_attn":
            drugBatch = self.cross_attn_module(drugBatch, targetBatch, drug_padding_mask, target_padding_mask)
            drugBatch = drugBatch.squeeze(1)  # bs * 512
            targetBatch = targetBatch.mean(1)
        elif self.atten_type == 'wsam':
            drugBatch = self.drugWeightedSumAndMax(drugBatch, drug_padding_mask)
            targetBatch = self.targetWeightedSumAndMax(targetBatch, target_padding_mask)
        else:
            # bz * token *dim
            drugBatch = drugBatch.sum(1)
            targetBatch = targetBatch.sum(1)
            # drugBatch = drugBatch.sum(1)
            # targetBatch = targetBatch.sum(1)
            # drugBatch = self.DrugModalityNormalization(drugBatch)
            # targetBatch = self.TargetModalityNormalization(targetBatch)

        jointBatch = torch.cat([drugBatch, targetBatch], dim=1)
        jointBatch = self.outputMLP(jointBatch)
        if self.focal_loss:
            jointBatch = jointBatch.squeeze(-1)
        else:
            pass

        return jointBatch
