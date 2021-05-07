import pickle as pkl
import pysmiles
import logging
import json

from utils.parser import *

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
MAX_NODES = 0

if __name__ == '__main__':
    args = parse_args()
    durgs = pkl.load(open(args.data_dir + '/drug.pkl', 'rb'))

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

    with open(args.data_dir + 'element.json', 'w') as f:
        f.write(json.dumps(element))
    with open(args.data_dir + 'hcount.json', 'w') as f:
        f.write(json.dumps(hcount))
