import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py

from models.dt_net import DTNet
from data.Dataset import DrugTargetInteractionDataset
from data.datautils import collate_fn
from utils.general import extract_logits
from utils.parser import *


def main():
    # load args
    args = parse_args()
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

    valData = DrugTargetInteractionDataset("val_full", args.neg_rate, args.step_size, args.target_h5_dir, args.freeze_protein_embedding,
                                           edge_weight=not args.no_edge_weight, use_hcount=not args.no_hcount)
    valLoader = DataLoader(valData, batch_size=args.ex_batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)

    # declaring the model, optimizer, scheduler and the loss function
    model = DTNet(args.freeze_protein_embedding, args.d_model, args.graph_layer, valData.drug_dataset.embedding_dim, args.mlp_depth,
                  args.graph_depth, args.GAT_head, args.target_in_size, args.pretrain_dir, args.gpu_id, args.atten_type, args.drug_conv,
                  args.target_conv, args.conv_dropout, args.add_transformer, args.focal_loss)
    if args.weight is not None:
        model.load_state_dict(torch.load(args.weight, map_location="cpu"))
        print("\nLoad model %s \n" % args.weight)
    model.to(device)

    f = h5py.File(args.logits_h5_dir, "w")
    dt = h5py.vlen_dtype(np.dtype('float32'))
    logits = f.create_dataset('logits', (len(valLoader),), dtype=dt)
    extract_logits(model, valLoader, logits, device)
    f.close()

    print("\nExtract Done.\n")
    return


if __name__ == "__main__":
    main()
