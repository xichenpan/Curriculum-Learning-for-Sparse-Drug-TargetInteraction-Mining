import torch
from torch.utils.data import Dataset
import pickle as pkl
import pysmiles
import json
import numpy as np
import logging
import h5py
import sys

sys.path.append("..")
from src.alphabets import Uniprot21
from src.models.embedding import *

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
logger = logging.getLogger('Data')

MAX_NODE_SIZE = 629


class DrugDataset(Dataset):
    def __init__(self, edge_weight=True, use_hcount=True, **kwargs):
        """
        :arg
            edge_weight:
                False - returns binary matrix, 1 for adjacent, 0 for not.
                True - use edge order to weight the matrix
            use_hcount:
                False - only use element of each node to embed node
                True - add extra hydrogens count info
        """
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
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smile_str = self.data[index]
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

        drug = (torch.from_numpy(node_embedding), torch.from_numpy(adjacent_matrix), torch.from_numpy(padding_mask))
        return drug


class TargetDataset(Dataset):
    def __init__(self, target_h5_dir, freeze_protein_embedding, **kwargs):
        super(TargetDataset, self).__init__()
        self.freeze_protein_embedding = freeze_protein_embedding
        if freeze_protein_embedding:
            self.h5 = target_h5_dir
        self.data = pkl.load(open("./data/target.pkl", 'rb'))
        self.alphabet = Uniprot21()
        print('Load Target Dataset Complete')
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.freeze_protein_embedding:
            with h5py.File(self.h5, "r") as f:
                x = np.array(f['target'][index]).reshape(-1, 121)
            return torch.tensor(x)
        else:
            protein_string = self.data[index]
            x = bytes(protein_string, encoding='utf8')
            x = x.upper()
            # convert to alphabet index
            x = self.alphabet.encode(x)
            x = torch.from_numpy(x)
            return x


class DrugTargetInteractionDataset(Dataset):
    def __init__(self, dataset, neg_rate, target_h5_dir, freeze_protein_embedding, **kwargs):
        super(DrugTargetInteractionDataset, self).__init__()
        if dataset == 'val_full':
            pkl_name = 'val'
        else:
            pkl_name = dataset

        self.pos_pairs = pkl.load(open('./data/%s_pos_pairs.pkl' % pkl_name, 'rb'))
        self.neg_pairs = pkl.load(open('./data/%s_neg_pairs.pkl' % pkl_name, 'rb'))
        self.dataset = dataset
        self.neg_rate = neg_rate
        self.drug_dataset = DrugDataset(**kwargs)
        self.target_dataset = TargetDataset(target_h5_dir, freeze_protein_embedding, **kwargs)
        print('Load DTI Dataset Complete')
        print('# %s pos pairs = %d' % (self.dataset, len(self.pos_pairs)))
        print('# %s neg pairs = %d' % (self.dataset, len(self.neg_pairs)))

    def __getitem__(self, index):
        # if self.dataset == "train":
        # index goes from 0 to stepSize-1
        # dividing the dataset into partitions of size equal to stepSize and selecting a random partition
        # fetch the sample at position 'index' in this randomly selected partition
        if index >= len(self.pos_pairs):
            if self.dataset in ['val', 'val_full']:
                index = index - len(self.pos_pairs)
            else:
                index = np.random.randint(0, len(self.neg_pairs))
            drug_idx, target_idx, label = self.neg_pairs[index][:]
        else:
            drug_idx, target_idx, label = self.pos_pairs[index][:]

        return self.drug_dataset[drug_idx], self.target_dataset[target_idx], label

    def __len__(self):
        if self.dataset == 'val':
            return 2 * len(self.pos_pairs)
        elif self.dataset == 'val_full':
            return len(self.pos_pairs) + len(self.neg_pairs)
        else:
            return (1 + self.neg_rate) * len(self.pos_pairs)
