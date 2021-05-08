import pickle as pkl
import pysmiles
import logging
import json

import sys

sys.path.append("..")
from utils.parser import *

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


def make_graph_dict(dataset):
    MAX_NODES = 0
    args = parse_args()
    durgs = pkl.load(open("../" + args.data_dir + "/" + dataset + '/drug.pkl', 'rb'))

    element = set()
    hcount = set()
    for drug in durgs:
        graph = pysmiles.read_smiles(drug)
        num_node = max([idx for idx, _ in graph.nodes.data()]) + 1
        MAX_NODES = max(MAX_NODES, num_node)

        for node_id, node_info in graph.nodes.data():
            element.add(node_info['element'])
            hcount.add(node_info['hcount'])

    print('MAX NODE SIZE = %d' % MAX_NODES)

    element = sorted(list(element))
    hcount = sorted(list(hcount))

    with open("../" + args.data_dir + "/" + dataset + 'element.json', 'w') as f:
        f.write(json.dumps(element))
    with open("../" + args.data_dir + "/" + dataset + 'hcount.json', 'w') as f:
        f.write(json.dumps(hcount))


if __name__ == '__main__':
    make_graph_dict("train")
    make_graph_dict("val")
