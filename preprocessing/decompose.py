import pandas as pd
import pickle
# from utils.parser import *
import sys
import numpy as np
import random

if __name__ == '__main__':
    data = pd.read_csv('../data/train.csv').values.tolist()
    drug, target, label = list(zip(*data))

    drug_set = sorted(list(set(drug)))
    target_set = sorted(list(set(target)))

    drug2idx = {
        drug: idx for idx, drug in enumerate(drug_set)
    }
    target2idx = {
        target: idx for idx, target in enumerate(target_set)
    }

    with open('../data/drug.pkl', 'wb') as f:
        pickle.dump(drug_set, f)

    with open('../data/target.pkl', 'wb') as f:
        pickle.dump(target_set, f)

    pairs = []
    for drug, target, label in data:
        drug_id = drug2idx[drug]
        target_id = target2idx[target]
        pairs.append([drug_id, target_id, label])

    np.random.seed(0)
    isTrain_mask = np.random.binomial(n=1, p=0.9, size=len(pairs))

    train_pos_pairs = []
    train_neg_pairs = []
    val_pos_pairs = []
    val_neg_pairs = []

    np.random.shuffle(pairs)

    for i in range(len(pairs)):
        pairs[i].append(isTrain_mask[i])
        if pairs[i][2] == 1:
            if isTrain_mask[i] == 1:
                train_pos_pairs.append(pairs[i][:-1])
            else:
                val_pos_pairs.append(pairs[i][:-1])
        else:
            if isTrain_mask[i] == 1:
                train_neg_pairs.append(pairs[i][:-1])
            else:
                val_neg_pairs.append(pairs[i][:-1])

    with open('../data/pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
        print('# Total = %d' % len(pairs))

    with open('../data/train_pos_pairs.pkl', 'wb') as f:
        pickle.dump(train_pos_pairs, f)
        print('# TrainPositive = %d' % len(train_pos_pairs))

    with open('../data/train_neg_pairs.pkl', 'wb') as f:
        pickle.dump(train_neg_pairs, f)
        print('# TrainNegative = %d' % len(train_neg_pairs))

    with open('../data/val_pos_pairs.pkl', 'wb') as f:
        pickle.dump(val_pos_pairs, f)
        print('# TestPositive = %d' % len(val_pos_pairs))

    with open('../data/val_neg_pairs.pkl', 'wb') as f:
        pickle.dump(val_neg_pairs, f)
        print('# TestNegative = %d' % len(val_neg_pairs))
