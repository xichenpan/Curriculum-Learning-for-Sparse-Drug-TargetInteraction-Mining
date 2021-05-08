import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil

from models.dt_net import DTNet
from data.Dataset import DrugTargetInteractionDataset
from data.datautils import collate_fn
from utils.general import num_params, train, evaluate
from utils.parser import *


def main():
    # load args
    args = parse_args()
    # not show curve
    matplotlib.use("Agg")
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
    trainData = DrugTargetInteractionDataset(
        "train",
        args.data_dir,
        args.step_size,
        args.pretrain_dir,
        device,
        edge_weight=not args.no_edge_weight,
        use_hcount=not args.no_hcount
    )
    trainLoader = DataLoader(
        trainData,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        **kwargs
    )

    valData = DrugTargetInteractionDataset(
        "val",
        args.data_dir,
        args.step_size,
        args.pretrain_dir,
        device,
        edge_weight=not args.no_edge_weight,
        use_hcount=not args.no_hcount
    )
    valLoader = DataLoader(
        valData,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        **kwargs
    )

    # declaring the model, optimizer, scheduler and the loss function

    model = DTNet(args.d_model, args.graph_layer, trainData.drug_dataset.embedding_dim, args.mlp_depth,
                  args.graph_depth, args.GAT_head, args.target_in_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr, betas=(args.MOMENTUM1, args.MOMENTUM2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.LR_SCHEDULER_FACTOR,
                                                     patience=args.LR_SCHEDULER_WAIT,
                                                     threshold=args.LR_SCHEDULER_THRESH,
                                                     threshold_mode="abs", min_lr=args.final_lr, verbose=True)
    loss_function = nn.CrossEntropyLoss()

    shutil.rmtree(args["CODE_DIRECTORY"] + "checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "checkpoints/models")
    os.mkdir(args["CODE_DIRECTORY"] + "checkpoints/plots")

    trainingLossCurve = list()
    validationLossCurve = list()
    trainingAccCurve = list()
    validationAccCurve = list()

    # printing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" % numTotalParams)
    print("Number of trainable parameters in the model = %d\n" % numTrainableParams)

    print("\nTraining the model .... \n")

    for step in range(args["NUM_STEPS"]):

        # train the model for one step
        trainingLoss, trainingAcc = train(model, trainLoader, optimizer, loss_function, device)
        trainingLossCurve.append(trainingLoss)
        trainingAccCurve.append(trainingAcc)

        # evaluate the model on validation set
        validationLoss, validationAcc = evaluate(model, valLoader, loss_function, device)
        validationLossCurve.append(validationLoss)
        validationAccCurve.append(validationAcc)

        # printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.Acc: %.3f  Val.Acc: %.3f"
              % (step, trainingLoss, validationLoss, trainingAcc, validationAcc))

        # make a scheduler step
        scheduler.step(validationAcc)

        # saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step % args.save_frequency == 0) or (step == args.num_steps - 1)) and (step != 0):
            savePath = args["CODE_DIRECTORY"] + "checkpoints/models/train-step_{:04d}-Acc_{:.3f}.pt".format(
                step, validationAcc)
            torch.save(model.state_dict(), savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Loss value")
            plt.plot(list(range(1, len(trainingLossCurve) + 1)),
                     trainingLossCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationLossCurve) + 1)),
                     validationLossCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] +
                        "checkpoints/plots/train-step_{:04d}-loss.png".format(step))
            plt.close()

            plt.figure()
            plt.title("Acc Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Acc")
            plt.plot(list(range(1, len(trainingAccCurve) + 1)),
                     trainingAccCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationAccCurve) + 1)),
                     validationAccCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] +
                        "checkpoints/plots/train-step_{:04d}-Acc.png".format(step))
            plt.close()

    print("\nTraining Done.\n")
    return


if __name__ == "__main__":
    main()
