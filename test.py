import numpy as np
import torch
from torch.utils.data import DataLoader
import csv

from data.Dataset import DrugTargetInteractionDataset
from data.datautils import collate_fn
from models.dt_net import DTNet
from utils.general import test
from utils.parser import *
from preprocessing.decompose import Decomposefull
from preprocessing.saveh5 import Saveh5


def main():
    # load args
    args = parse_args()
    assert args.weight is not None
    # Preprocessing
    Decomposefull(args.csv_file, '')
    Saveh5('')
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

    # declaring the train and test datasets and their corresponding dataloaders
    testData = DrugTargetInteractionDataset("test", args.neg_rate, args.step_size, args.target_h5_dir, args.freeze_protein_embedding,
                                            edge_weight=not args.no_edge_weight, use_hcount=not args.no_hcount)
    testLoader = DataLoader(testData, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)

    # declaring the model, optimizer, scheduler and the loss function
    model = DTNet(args.freeze_protein_embedding, args.d_model, args.graph_layer, testData.drug_dataset.embedding_dim, args.mlp_depth,
                  args.graph_depth, args.GAT_head, args.target_in_size, args.pretrain_dir, args.gpu_id, args.atten_type, args.drug_conv,
                  args.target_conv, args.conv_dropout, args.add_transformer, args.focal_loss)
    model.load_state_dict(torch.load(args.weight))
    model.to(device).eval()

    print("Evaluating the model from <== %s\n" % args.weight)
    with torch.no_grad():
        testTP, testFP, testFN, testTN, testAcc, testF1 = test(model, testLoader, args.threshold, device)

    with open("result.csv", 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["Accuracy", "F1 score"])
        csv_write.writerow(["{0:.3f}".format(testAcc), "{0:.3f}".format(testF1)])

    print("\nEvaluation Done.\n")
    return


if __name__ == "__main__":
    main()
