import torch
from tqdm import tqdm
import time


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def compute_score(outputBatch, labelinputBatch):
    pred = outputBatch.argmax(dim=1).long()

    TP = ((pred == 1).float() * (labelinputBatch == 1).float()).sum().item()
    FP = ((pred == 1).float() * (labelinputBatch == 0).float()).sum().item()
    FN = ((pred == 0).float() * (labelinputBatch == 1).float()).sum().item()
    TN = ((pred == 0).float() * (labelinputBatch == 0).float()).sum().item()

    acc = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = precision * recall * 2 / (precision + recall)
    return TP, FP, FN, TN, acc, F1


def train(model, trainLoader, optimizer, loss_function, device, writer, step):
    trainingLoss = 0
    outputAll = []
    labelinputAll = []
    # time_read_st = time.time()

    model.train()
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(
            tqdm(trainLoader, leave=False, desc="Train", ncols=75)):
        # print("TIME READ CMD: ", time.time() - time_read_st)

        druginputBatch = (
            druginputBatch[0].float().to(device), druginputBatch[1].float().to(device),
            druginputBatch[2].bool().to(device))
        targetinputBatch = (targetinputBatch[0].int().to(device), targetinputBatch[1].bool().to(device))
        labelinputBatch = labelinputBatch.long().to(device)

        optimizer.zero_grad()
        outputBatch = model(druginputBatch, targetinputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, labelinputBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        outputAll.append(outputBatch.detach().cpu())
        labelinputAll.append(labelinputBatch.cpu())

        writer.add_scalar('cross_entropy_loss', loss.item(), step * len(trainLoader))

    outputAll = torch.cat(outputAll, 0)
    labelinputAll = torch.cat(labelinputAll, 0)
    trainingLoss = trainingLoss / len(trainLoader)
    TP, FP, FN, TN, acc, F1 = compute_score(outputAll, labelinputAll)
    return trainingLoss, TP, FP, FN, TN, acc, F1


def evaluate(model, evalLoader, loss_function, device):
    evalLoss = 0
    outputAll = []
    labelinputAll = []
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(
            tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        druginputBatch = (
            druginputBatch[0].float().to(device), druginputBatch[1].float().to(device),
            druginputBatch[2].bool().to(device))
        targetinputBatch = (targetinputBatch[0].int().to(device), targetinputBatch[1].bool().to(device))
        labelinputBatch = labelinputBatch.long().to(device)

        model.eval()
        with torch.no_grad():
            outputBatch = model(druginputBatch, targetinputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, labelinputBatch)

        evalLoss = evalLoss + loss.item()
        outputAll.append(outputBatch.detach().cpu())
        labelinputAll.append(labelinputBatch.cpu())

    outputAll = torch.cat(outputAll, 0)
    labelinputAll = torch.cat(labelinputAll, 0)
    evalLoss = evalLoss / len(evalLoader.dataset)
    TP, FP, FN, TN, acc, F1 = compute_score(outputAll, labelinputAll)
    return evalLoss, TP, FP, FN, TN, acc, F1
