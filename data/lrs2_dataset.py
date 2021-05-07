from torch.utils.data import Dataset
import numpy as np
import h5py

from .utils import prepare_pretrain_input
from .utils import prepare_main_input


class LRS2Pretrain(Dataset):
    """
    A custom dataset class for the LRS2 pretrain (includes pretain, preval) dataset.
    """

    def __init__(self, dataset, datadir, h5dir, numWords, charToIx, stepSize):
        super(LRS2Pretrain, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/pretrain/" + line.strip() for line in lines]
        self.h5 = h5py.File(h5dir, "r")
        self.numWords = numWords
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        return

    def __getitem__(self, index):
        if self.dataset == "pretrain":
            # index goes from 0 to stepSize-1
            # dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            # fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)

        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        targetFile = self.datalist[index] + ".txt"
        inp, trgt, trgtLen = prepare_pretrain_input(index, self.h5, self.dataset, targetFile, self.numWords, self.charToIx)
        return inp, trgt, trgtLen

    def __len__(self):
        # each iteration covers only a random subset of all the training samples whose size is given by the step size
        # this is done only for the pretrain set, while the whole preval set is considered
        if self.dataset == "pretrain":
            return self.stepSize
        else:
            return len(self.datalist)


class LRS2Main(Dataset):
    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, h5dir, charToIx, stepSize):
        super(LRS2Main, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/main/" + line.strip().split(" ")[0] for line in lines]
        self.h5 = h5py.File(h5dir, "r")
        self.charToIx = charToIx  # character to index mapping
        self.dataset = dataset
        # number of samples in one step (virtual epoch)
        self.stepSize = stepSize
        return

    def __getitem__(self, index):
        # using the same procedure as in pretrain dataset class only for the train dataset
        if self.dataset == "train":
            # [    0, 16384, 32768]
            base = self.stepSize * np.arange(int(len(self.datalist) / self.stepSize) + 1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]  # len = 45839
            index = ixs[0] if len(ixs) == 1 else np.random.choice(ixs)

        # passing the sample files and the target file paths to the prepare function to obtain the input tensors
        targetFile = self.datalist[index] + ".txt"
        inp, trgt, trgtLen = prepare_main_input(index, self.h5, self.dataset, targetFile, self.charToIx)
        return inp, trgt, trgtLen

    def __len__(self):
        # using step size only for train dataset and not for val and test datasets because
        # the size of val and test datasets is smaller than step size and we generally want to validate and test
        # on the complete dataset
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)
