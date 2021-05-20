import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    # drug
    drug_node_embedding = [data[0][0] for data in dataBatch]
    drug_adjacent_matrix = [data[0][1] for data in dataBatch]
    drug_padding_mask = [data[0][2] for data in dataBatch]
    druginputBatch = (drug_node_embedding, drug_adjacent_matrix, drug_padding_mask)

    # target
    target_embedding = [data[1] for data in dataBatch]
    target_padding_mask = torch.zeros((len(target_embedding), len(max(target_embedding, key=len))), dtype=torch.bool)
    for i, seq in enumerate(target_embedding):
        target_padding_mask[i, len(seq):] = True
    targetinputBatch = (pad_sequence(target_embedding, batch_first=True), target_padding_mask)

    # label
    labelinputBatch = [data[2] for data in dataBatch]

    return druginputBatch, targetinputBatch, labelinputBatch
