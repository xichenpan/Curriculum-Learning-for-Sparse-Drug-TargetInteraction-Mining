import numpy as np
import h5py
import pickle as pkl
from tqdm import tqdm

import sys

sys.path.append("..")
from utils.parser import parse_args
from src.alphabets import Uniprot21
from utils.protein_embedding import *


def fill_in_data(data, alphabet, target, target_net, device):
    for i, item in enumerate(tqdm(data, leave=False, desc="Saving hdf5", ncols=75)):
        inp = bytes(item, encoding='utf8')
        inp = inp.upper()
        inp = alphabet.encode(inp)
        inp = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            inp = embedding(inp.to(torch.int64), target_net, device)
        target[i] = inp.cpu().numpy().flatten()


def Saveh5(relevantdir):
    args = parse_args()

    torch.cuda.set_device(args.gpu_id)
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")

    lm = BiLM(nin=22, embedding_dim=21, hidden_dim=1024, num_layers=2, nout=21)
    model_ = StackedRNN(nin=21, nembed=512, nunits=512, nout=100, nlayers=3, padding_idx=20, dropout=0, lm=lm)
    model = OrdinalRegression(embedding=model_, n_classes=5)
    state = torch.load(relevantdir + args.pretrain_dir)
    model.load_state_dict(state)
    target_net = load_model(model, device=args.gpu_id)

    f = h5py.File(relevantdir + args.target_h5_dir, "w")
    data = pkl.load(open(relevantdir + "data/target.pkl", 'rb'))
    alphabet = Uniprot21()
    dt = h5py.vlen_dtype(np.dtype('float32'))
    target = f.create_dataset('target', (len(data),), dtype=dt)
    fill_in_data(data, alphabet, target, target_net, device)
    f.close()


if __name__ == "__main__":
    Saveh5('../')
