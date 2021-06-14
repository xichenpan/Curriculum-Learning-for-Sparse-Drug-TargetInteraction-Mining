import torch
from tqdm import tqdm


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def compute_score(outputBatch, labelinputBatch, neg_rate):
    try:
        pred = outputBatch.argmax(dim=1).long()
    except:
        pred = outputBatch.long()
    if neg_rate == -1:
        scale_times = 1
    else:
        scale_times = 0.3 * neg_rate

    TP = ((pred == 1).float() * (labelinputBatch == 1).float()).sum().item()
    FP = ((pred == 1).float() * (labelinputBatch == 0).float()).sum().item() * scale_times
    FN = ((pred == 0).float() * (labelinputBatch == 1).float()).sum().item()
    TN = ((pred == 0).float() * (labelinputBatch == 0).float()).sum().item() * scale_times

    try:
        acc = (TP + TN) / (TP + FP + FN + TN)
    except:
        acc = -1
    try:
        precision = TP / (TP + FP)
    except:
        precision = -1
    try:
        recall = TP / (TP + FN)
    except:
        recall = -1
    try:
        F1 = precision * recall * 2 / (precision + recall)
    except:
        F1 = -1
    return TP, FP, FN, TN, acc, F1


def train(model, trainLoader, optimizer, loss_function, device, writer, step, neg_rate):
    trainingLoss = 0
    outputAll = []
    labelinputAll = []

    model.train()
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train", ncols=75)):
        druginputBatch = (druginputBatch[0].float().to(device), druginputBatch[1].float().to(device), druginputBatch[2].bool().to(device))
        targetinputBatch = (targetinputBatch[0].to(device), targetinputBatch[1].bool().to(device))
        labelinputBatch = labelinputBatch.float().to(device)

        optimizer.zero_grad()
        outputBatch = model(druginputBatch, targetinputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, labelinputBatch, reduction="mean")
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        outputAll.append(outputBatch.detach().cpu())
        labelinputAll.append(labelinputBatch.cpu())
        writer.add_scalar('cross_entropy_loss', loss.item(), step * len(trainLoader))

    outputAll = torch.cat(outputAll, 0)
    labelinputAll = torch.cat(labelinputAll, 0)
    trainingLoss = trainingLoss / len(trainLoader)
    TP, FP, FN, TN, acc, F1 = compute_score(outputAll, labelinputAll, -1)
    return trainingLoss, TP, FP, FN, TN, acc, F1


def evaluate(model, evalLoader, loss_function, device, neg_rate):
    evalLoss = 0
    outputAll = []
    labelinputAll = []

    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval", ncols=75)):
        druginputBatch = (druginputBatch[0].float().to(device), druginputBatch[1].float().to(device), druginputBatch[2].bool().to(device))
        targetinputBatch = (targetinputBatch[0].to(device), targetinputBatch[1].bool().to(device))
        labelinputBatch = labelinputBatch.float().to(device)

        model.eval()
        with torch.no_grad():
            outputBatch = model(druginputBatch, targetinputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, labelinputBatch, reduction="mean")

        evalLoss = evalLoss + loss.item()
        outputAll.append(outputBatch.detach().cpu())
        labelinputAll.append(labelinputBatch.cpu())

    outputAll = torch.cat(outputAll, 0)
    labelinputAll = torch.cat(labelinputAll, 0)
    evalLoss = evalLoss / len(evalLoader)
    TP, FP, FN, TN, acc, F1 = compute_score(outputAll, labelinputAll, neg_rate)
    return evalLoss, TP, FP, FN, TN, acc, F1


def extract_logits(model, evalLoader, logits, device):
    for batch, (druginputBatch, targetinputBatch, labelinputBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Extract", ncols=75)):
        druginputBatch = (druginputBatch[0].float().to(device), druginputBatch[1].float().to(device), druginputBatch[2].bool().to(device))
        targetinputBatch = (targetinputBatch[0].to(device), targetinputBatch[1].bool().to(device))

        model.eval()
        with torch.no_grad():
            outputBatch = model(druginputBatch, targetinputBatch)

        logits[batch] = outputBatch.cpu().numpy().flatten()
