import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    # drug
    if dataBatch[0][0] is not None:
        drug_node_embedding = torch.stack([data[0][0] for data in dataBatch])
        drug_adjacent_matrix = torch.stack([data[0][1] for data in dataBatch])
        drug_padding_mask = torch.stack([data[0][2] for data in dataBatch])
        druginputBatch = (drug_node_embedding, drug_adjacent_matrix, drug_padding_mask)
    else:
        druginputBatch = None

    # target
    if dataBatch[0][1] is not None:
        target_embedding = [data[1] for data in dataBatch]
        target_padding_mask = torch.zeros((len(target_embedding), len(max(target_embedding, key=len))), dtype=torch.bool)
        for i, seq in enumerate(target_embedding):
            target_padding_mask[i, len(seq):] = True
        targetinputBatch = (pad_sequence(target_embedding, batch_first=True), target_padding_mask)
    else:
        targetinputBatch = None

    # label
    if dataBatch[0][2] is not None:
        labelinputBatch = torch.tensor([data[2] for data in dataBatch])
    else:
        labelinputBatch = None

    return druginputBatch, targetinputBatch, labelinputBatch
