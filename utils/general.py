import torch
from tqdm import tqdm


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def compute_Acc(outputBatch, labelinputBatch):
    return 1


def train(model, trainLoader, optimizer, loss_function, device):
    trainingLoss = 0
    trainingAcc = 0
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(
            tqdm(trainLoader, leave=False, desc="Train", ncols=75)):
        druginputBatch = (druginputBatch[0].float().to(device), druginputBatch[1].float().to(device),
                          druginputBatch[2].float().to(device))
        targetinputBatch = None
        labelinputBatch = labelinputBatch.int().to(device)

        optimizer.zero_grad()
        model.train()
        outputBatch = model(druginputBatch, targetinputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, labelinputBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        trainingAcc = trainingAcc + compute_Acc(outputBatch, labelinputBatch)

    trainingLoss = trainingLoss / len(trainLoader)
    trainingAcc = trainingAcc / len(trainLoader)
    return trainingLoss, trainingAcc


def evaluate(model, evalLoader, loss_function, device):
    evalLoss = 0
    evalAcc = 0
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(
            tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        druginputBatch = (druginputBatch[0].float().to(device), druginputBatch[1].float().to(device),
                          druginputBatch[2].float().to(device))
        targetinputBatch = None
        labelinputBatch = labelinputBatch.int().to(device)

        model.eval()
        with torch.no_grad():
            outputBatch = model(druginputBatch, targetinputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, labelinputBatch)

        evalLoss = evalLoss + loss.item()
        evalAcc = evalAcc + compute_Acc(outputBatch, labelinputBatch)

    evalLoss = evalLoss / len(evalLoader)
    evalAcc = evalAcc / len(evalLoader)
    return evalLoss, evalAcc
