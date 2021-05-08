import pandas as pd
import pickle

import sys

sys.path.append("..")
from utils.parser import *


def decompose(dataset):
    args = parse_args()
    data = pd.read_csv(".." + args.data_dir + "/" + dataset + "/" + dataset + ".csv").values.tolist()
    drug, target, label = list(zip(*data))

    drug_set = list(set(drug))
    target_set = list(set(target))

    drug2idx = {
        drug: idx for idx, drug in enumerate(drug_set)
    }
    target2idx = {
        target: idx for idx, target in enumerate(target_set)
    }

    with open(".." + args.data_dir + "/" + dataset + '/drug.pkl', 'wb') as f:
        pickle.dump(drug_set, f)

    with open(".." + args.data_dir + "/" + dataset + '/target.pkl', 'wb') as f:
        pickle.dump(target_set, f)

    pairs = []

    for drug, target, label in data:
        drug_id = drug2idx[drug]
        target_id = target2idx[target]
        pairs.append([drug_id, target_id, label])

    with open(".." + args.data_dir + "/" + dataset + '/pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)


if __name__ == '__main__':
    decompose("train")
    decompose("val")
