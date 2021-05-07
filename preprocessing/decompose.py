import pandas as pd
import numpy as np
import time
import pickle

if __name__ == '__main__':
    data = pd.read_csv('train.csv').values.tolist()
    drug, target, label = list(zip(*data))

    drug_set = list(set(drug))
    target_set = list(set(target))

    drug2idx = {
        drug: idx for idx, drug in enumerate(drug_set)
    }
    target2idx = {
        target: idx for idx, target in enumerate(target_set)
    }

    with open('drug.pkl', 'wb') as f:
        pickle.dump(drug_set, f)

    with open('target.pkl', 'wb') as f:
        pickle.dump(target_set, f)

    pairs = []

    for drug, target, label in data:
        drug_id = drug2idx[drug]
        target_id = target2idx[target]
        pairs.append([drug_id, target_id, label])

    with open('pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
