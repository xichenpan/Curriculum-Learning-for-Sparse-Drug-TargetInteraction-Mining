import torch
from torch.utils.data import DataLoader
import numpy as np
import h5py
import logging

from models.dt_net import DTNet
from data.Dataset import DrugTargetInteractionDataset
from data.datautils import collate_fn
from utils.general import find_threshold
from utils.parser import *


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', filename='res.log', filemode='w')
    logger = logging.getLogger(__name__)
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

    valData = DrugTargetInteractionDataset("find_full", args.neg_rate, args.step_size, args.target_h5_dir, args.freeze_protein_embedding,
                                           edge_weight=not args.no_edge_weight, use_hcount=not args.no_hcount)
    valLoader = DataLoader(valData, batch_size=args.ex_batch_size, collate_fn=collate_fn, shuffle=False, **kwargs)

    threshold_list = [0.9]

    f = h5py.File(args.logits_h5_dir, "w")
    bestF1 = -1
    bestthreshold = -1
    for threshold in threshold_list:
        TP, FP, FN, TN, acc, F1 = find_threshold(valLoader, f, threshold, device)
        logger.info("%s Result: TP: %d || FP: %d || FN: %d || TN: %d || acc: %.3f || F1: %.3f" % (threshold, TP, FP, FN, TN, acc, F1))
        if F1 > bestF1:
            bestF1 = F1
            bestthreshold = threshold

    logger.info("Find Done.\nBest Result: threshold: %s || F1: %.3f" % (bestthreshold, bestF1))
    f.close()

    return


if __name__ == "__main__":
    main()
