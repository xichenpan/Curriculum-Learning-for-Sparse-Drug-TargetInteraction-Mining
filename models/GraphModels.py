import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolutionLayer(nn.Module):
    def __init__(self, dim, **kwargs):
        super(GraphConvolutionLayer, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )

    def forward(self, node_embedding, adj=None, norm_adj=None):
        if norm_adj is None:
            norm_adj = adj + torch.eye(adj.shape[-1]).to(adj.device)
            inv_sqrt_diag = norm_adj.diagonal(dim1=-2, dim2=-1).diag_embed().pow(-0.5)
            norm_adj = torch.bmm(torch.bmm(inv_sqrt_diag, norm_adj), inv_sqrt_diag)

        return self.feedforward(torch.bmm(norm_adj, node_embedding))


class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, head=1, **kwargs):
        super(GraphAttentionLayer, self).__init__()
        self.h = head
        self.feedforward = nn.ModuleList()
        self.attention_vec = nn.Parameter(torch.randn(head, 2 * dim))
        for _ in range(head):
            self.feedforward.append(nn.Linear(dim, dim))
        self.relu = nn.ReLU()

    def forward(self, node_embedding, adj=None, **kwargs):
        '''
            TODO: sparse attention
        '''
        out = []
        num_node = node_embedding.shape[1]
        for i, FC in enumerate(self.feedforward):
            embed = FC(node_embedding)  # b * N * dim
            embed_repeat1 = embed.unsqueeze(1).repeat(1, num_node, 1, 1)  # b * N * N * dim
            embed_repeat2 = embed.unsqueeze(2).repeat(1, 1, num_node, 1)  # b * N * N * dim
            embed_cat = torch.cat([embed_repeat1, embed_repeat2], -1)  # b * N * N * 2dim
            vec = self.attention_vec[i]
            attention = (embed_cat * vec).sum(-1) * adj  # b * N * N
            attention = F.softmax(F.leaky_relu(attention), -1)
            out.append(torch.bmm(attention, embed))

        out = torch.stack(out, 0).mean(0)
        return self.relu(out)


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batchnorm, **kwargs):
        super(MLPLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
        if batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.layer(x)
        if 'bn' in self.__dir__():
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        return x


class GraphNeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, layer_type='GCN', num_pre=1, num_graph_layer=1, batchnorm=True, head=1,
                 **kwargs):
        '''
        :param in_dim: input dimensionality
        :param out_dim: output dimensionality, as well as all hidden feature
        :param batchnorm: whether to use BN in pre-MLP
        :param num_pre: number of pre-MLP layers
        :param layer_type: GCN or GAT
        :param num_graph_layer: number of graph model layers
        :param head: attention head in GAT
        :param kwargs:
        '''
        super(GraphNeuralNetwork, self).__init__()
        self.PreMLP = nn.ModuleList(
            [MLPLayer(in_dim, out_dim, batchnorm)]
            + [MLPLayer(out_dim, out_dim, batchnorm) for _ in range(num_pre - 1)]
        )
        if layer_type == 'GCN':
            layer = GraphConvolutionLayer
        elif layer_type == 'GAT':
            layer = GraphAttentionLayer
        else:
            raise NotImplementedError

        self.GraphLayers = nn.ModuleList([layer(out_dim, head=head) for _ in range(num_graph_layer)])

    def forward(self, node_embedding, adj):
        norm_adj = adj + torch.eye(adj.shape[-1]).to(adj.device)
        inv_sqrt_diag = norm_adj.diagonal(dim1=-2, dim2=-1).diag_embed().pow(-0.5)
        norm_adj = torch.bmm(torch.bmm(inv_sqrt_diag, norm_adj), inv_sqrt_diag)

        for module in self.PreMLP:
            node_embedding = module(node_embedding)

        for module in self.GraphLayers:
            node_embedding = module(node_embedding, adj=adj, norm_adj=norm_adj)

        return node_embedding
