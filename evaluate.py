import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.Dataset import DrugTargetInteractionDataset
from data.datautils import collate_fn
from models.dt_net import DTNet
from utils.general import evaluate
from utils.parser import *


def main():
    # load args
    args = parse_args()
    assert args.weight is not None
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # check device
    torch.cuda.set_device(args.gpu_id)
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args.num_workers, "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # declaring the train and validation datasets and their corresponding dataloaders

    valData = DrugTargetInteractionDataset("val_full", args.neg_rate, args.target_h5_dir, args.freeze_protein_embedding,
                                           edge_weight=not args.no_edge_weight, use_hcount=not args.no_hcount)
    valLoader = DataLoader(valData, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)

    # declaring the model, optimizer, scheduler and the loss function
    model = DTNet(args.freeze_protein_embedding, args.d_model, args.graph_layer, valData.drug_dataset.embedding_dim, args.mlp_depth, args.graph_depth,
                  args.GAT_head, args.target_in_size, args.pretrain_dir, args.gpu_id, args.atten_type, args.drug_conv, args.target_conv,
                  args.conv_dropout, args.add_transformer)
    model.load_state_dict(torch.load(args.weight))
    model.to(device).eval()

    loss_function = nn.CrossEntropyLoss()

    print("Evaluating the model from <== %s\n" % args.weight)
    with torch.no_grad():
        valLoss, valTP, valFP, valFN, valTN, valAcc, valF1 = evaluate(model, valLoader, loss_function, device)

    print("Val|| Loss: %.6f || Acc: %.3f  F1: %.3f || TP: %d TN %d FP: %d FN: %d" % (valLoss, valAcc, valF1, valTP, valTN, valFP, valFN))

    print("\nEvaluation Done.\n")
    return


if __name__ == "__main__":
    main()
