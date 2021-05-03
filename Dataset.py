import torch
import pickle as pkl
import pysmiles
import json
import numpy as np
import logging

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
logger = logging.getLogger('Data')

MAX_NODE_SIZE = 629


class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, edge_weight=True, use_hcount=True, **kwargs):
        '''
        :arg
            edge_weight:
                False - returns binary matrix, 1 for adjacent, 0 for not.
                True - use edge order to weight the matrix
            use_hcount:
                False - only use element of each node to embed node
                True - add extra hydrogens count info
        '''
        super(DrugDataset, self).__init__()
        self.edge_weight = edge_weight
        self.use_hcount = use_hcount
        self.data = pkl.load(open('./data/drug.pkl', 'rb'))

        element = json.load(open('./data/element.json'))
        hcount = json.load(open('./data/hcount.json'))

        self.element2idx = {ele: i for i, ele in enumerate(element)}
        self.hcount2idx = {count: i for i, count in enumerate(hcount)}

        if use_hcount:
            self.embedding_dim = len(self.element2idx) + len(self.hcount2idx)
        else:
            self.embedding_dim = len(self.element2idx)

        print('Load Drug Dataset Complete')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smile_str = self.data[item]
        graph = pysmiles.read_smiles(smile_str)

        node_embedding = np.zeros([MAX_NODE_SIZE, self.embedding_dim])
        for node_id, node_info in graph.nodes.data():
            element = node_info['element']
            hcount = node_info['hcount']
            node_embedding[node_id, self.element2idx[element]] = 1
            if self.use_hcount:
                node_embedding[node_id, self.hcount2idx[hcount]] = 1

        padding_mask = (node_embedding.sum(1) == 0).astype(float)

        adjacent_matrix = np.zeros([MAX_NODE_SIZE, MAX_NODE_SIZE])
        for a, b, edge_info in graph.edges.data():
            val = edge_info['order'] if self.edge_weight else 1
            adjacent_matrix[a, b] = val
            adjacent_matrix[b, a] = val

        return node_embedding, adjacent_matrix, padding_mask


class TargetDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(TargetDataset, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class DrugTargetInteractionDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.drug_dataset = DrugDataset(**kwargs)
        self.target_dataset = TargetDataset(**kwargs)
        self.pairs = pkl.load(open('./data/pairs.pkl', 'rb'))
        print('Load DTI Dataset Complete')

    def __len__(self):
        return len(self.pairs)
        pass

    def __getitem__(self, item):
        drug_idx, target_idx, label = self.pairs[item]
        return self.drug_dataset[drug_idx], self.target_dataset[target_idx], label
