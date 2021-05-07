import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.special import softmax

from config import args


def prepare_main_input(index, h5, dataset, targetFile, charToIx):
    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

    if targetFile is not None:

        # reading the target from the target file and converting each character to its corresponding index
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]

        trgt = [charToIx[char] for char in trgt]
        trgt.append(charToIx["<EOS>"])
        trgt = np.array(trgt)
        trgtLen = len(trgt)

        # the target length must be less than or equal to 100 characters (restricted space where our model will work)
        if trgtLen > 100:
            print("Target length more than 100 characters. Exiting")
            exit()

    # audio file
    audInp = torch.from_numpy(np.array(h5[dataset + "_wav"][index]))

    # visual file
    vidInp = np.array(h5[dataset + "_png"][index]).reshape(-1, 2048)
    vidInp = torch.tensor(vidInp)

    inp = (audInp, vidInp)
    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
        trgtLen = torch.tensor(trgtLen)
    else:
        trgt, trgtLen = None, None

    return inp, trgt, trgtLen


def prepare_pretrain_input(index, h5, dataset, targetFile, numWords, charToIx):
    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    # reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    # if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        # audio file
        audInp = torch.from_numpy(np.array(h5[dataset + "_wav"][index]))

        # visual file
        vidInp = np.array(h5[dataset + "_png"][index]).reshape(-1, 2048)
        vidInp = torch.tensor(vidInp)

    else:
        # make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i + numWords])
                  for i in range(len(words) - numWords + 1)]
        nWordLens = np.array(
            [len(nWord) + 1 for nWord in nWords]).astype(np.float)

        # choose the sub-sequence for target according to a softmax distribution of the lengths
        # this way longer sub-sequences (which are more diverse) are selected more often while
        # the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        # reading the start and end times in the video corresponding to the selected sub-sequence
        startTime = float(lines[4 + ix].split(" ")[1])
        endTime = float(lines[4 + ix + numWords - 1].split(" ")[2])
        # loading the audio
        samplerate = args["AUDIO_SAMPLE_RATE"]
        audInp = torch.from_numpy(np.array(h5[dataset + "_wav"][index]))[int(samplerate * startTime):int(samplerate * endTime)]
        # loading visual features
        videoFPS = args["VIDEO_FPS"]
        vidInp = np.array(h5[dataset + "_png"][index]).reshape(-1, 2048)
        vidInp = torch.tensor(vidInp)
        vidInp = vidInp[int(np.floor(videoFPS * startTime)): int(np.ceil(videoFPS * endTime))]

    # converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]
    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)

    inp = (audInp, vidInp)
    trgt = torch.from_numpy(trgt)
    trgtLen = torch.tensor(trgtLen)

    return inp, trgt, trgtLen


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    # audio & mask
    aud_seq_list = [data[0][0] for data in dataBatch]
    aud_padding_mask = torch.zeros((len(aud_seq_list), len(max(aud_seq_list, key=len))), dtype=torch.bool)
    for i, seq in enumerate(aud_seq_list):
        aud_padding_mask[i, len(seq):] = True
    # visual & len
    vis_seq_list = torch.cat([data[0][1] for data in dataBatch])
    vis_len = torch.tensor([len(data[0][1]) for data in dataBatch])

    inputBatch = (pad_sequence(aud_seq_list, batch_first=True), aud_padding_mask, vis_seq_list, vis_len)

    if not any(data[1] is None for data in dataBatch):
        targetBatch = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch = None

    if not any(data[2] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[2] for data in dataBatch])
    else:
        targetLenBatch = None

    return inputBatch, targetBatch, targetLenBatch


def req_input_length(trgt, trgtLen):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    trgt_list = torch.split(trgt, trgtLen.tolist(), dim=0)
    reqLen = torch.zeros(len(trgtLen))
    for index, t in enumerate(trgt_list):
        reqLen[index] = len(t)
        lastChar = t[0]
        for i in range(1, len(t)):
            if t[i] != lastChar:
                lastChar = t[i]
            else:
                reqLen[index] = reqLen[index] + 1
    return reqLen
