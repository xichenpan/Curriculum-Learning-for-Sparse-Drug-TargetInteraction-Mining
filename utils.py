import pickle as pkl
import numpy as np
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bz', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--graph_layer', type=str, default='GCN', choices=['GCN', 'GAT'])
    parser.add_argument('--GAT_head', type=int, default=2)
    parser.add_argument('--graph_depth', type=int, default=2)
    parser.add_argument('--mlp_depth', type=int, default=2)
    parser.add_argument('--no_edge_weight', default=False, action='store_true')
    parser.add_argument('--no_hcount', default=False, action='store_true')
    args = parser.parse_args()
    print('============Args==============')
    print(args)
    return args
