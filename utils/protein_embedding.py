import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append("..")
from src.alphabets import Uniprot21
from src.models.sequence import *
from src.models.comparison import *
from src.models.embedding import *


def unstack_lstm(lstm):
    device = next(iter(lstm.parameters())).device

    in_size = lstm.input_size
    hidden_dim = lstm.hidden_size
    layers = []
    for i in range(lstm.num_layers):
        layer = nn.LSTM(in_size, hidden_dim, batch_first=True, bidirectional=True)
        layer.to(device)

        attributes = ['weight_ih_l', 'weight_hh_l', 'bias_ih_l', 'bias_hh_l']
        for attr in attributes:
            dest = attr + '0'
            src = attr + str(i)
            getattr(layer, dest).data[:] = getattr(lstm, src)
            # setattr(layer, dest, getattr(lstm, src))

            dest = attr + '0_reverse'
            src = attr + str(i) + '_reverse'
            getattr(layer, dest).data[:] = getattr(lstm, src)
            # setattr(layer, dest, getattr(lstm, src))
        layer.flatten_parameters()
        layers.append(layer)
        in_size = 2 * hidden_dim
    return layers


def embed_stack(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False):
    zs = []

    x_onehot = x.new(x.size(0), x.size(1), 21).float().zero_()
    x_onehot.scatter_(2, x.unsqueeze(2), 1)
    zs.append(x_onehot)

    h = lm_embed(x)
    if include_lm and not final_only:
        zs.append(h)

    if lstm_stack is not None:
        for lstm in lstm_stack:
            h, _ = lstm(h)
            if not final_only:
                zs.append(h)
        h = proj(h.squeeze(0)).unsqueeze(0)
        zs.append(h)

    z = torch.cat(zs, 2)
    return z


def embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=False
                   , pool='none', device=0):
    if len(x) == 0:
        return None

    alphabet = Uniprot21()
    x = x.upper()
    # convert to alphabet index
    x = alphabet.encode(x)
    x = torch.from_numpy(x)
    if isinstance(device, int):
        x = x.to(device)

    # embed the sequence
    with torch.no_grad():
        x = x.long().unsqueeze(0)
        z = embed_stack(x, lm_embed, lstm_stack, proj
                        , include_lm=include_lm, final_only=final_only)
        # pool if needed
        z = z.squeeze(0)
        if pool == 'sum':
            z = z.sum(0)
        elif pool == 'max':
            z, _ = z.max(0)
        elif pool == 'avg':
            z = z.mean(0)
        # z = z.cpu().numpy()
    return z


def load_model(model, device=0):
    encoder = model
    encoder.eval()

    if isinstance(device, int):
        encoder = encoder.to(device)

    encoder = encoder.embedding

    lm_embed = encoder.embed
    lstm_stack = unstack_lstm(encoder.rnn)
    proj = encoder.proj

    return lm_embed, lstm_stack, proj


def embedding(x, model, device):
    """
   :param x: input protein sequence : batch * length
   :param model: (lm_embed, lstm_stack, proj)
   :param device: GPU:0,1,2,3
   :return: z
   """
    lm_embed, lstm_stack, proj = model
    z = embed_sequence(x, lm_embed, lstm_stack, proj, include_lm=True, final_only=True, pool=None, device=device)
    return z


if __name__ == "__main__":
    """-----load model----"""
    root = "/home/htxue/data/3E-DrugTargetInteraction/"
    model_path = os.path.join(root, "pretrained-model/model_weight.bin")
    device = 0

    lm = BiLM(nin=22, embedding_dim=21, hidden_dim=1024, num_layers=2, nout=21)
    model_ = StackedRNN(nin=21, nembed=512, nunits=512, nout=100, nlayers=3, padding_idx=20, dropout=0, lm=lm)
    model = OrdinalRegression(embedding=model_, n_classes=5)

    print(model)

    tmp = torch.load(os.path.join(root, model_path))
    model.load_state_dict(tmp)
    """---------------"""

    model = load_model(model, device=device)  # decompose the model into three parts

    x = b'ABCDABDD'

    import time

    st = time.time()

    z = embedding(x, model, device)

    print(time.time() - st)
    print(z)
    print(z.shape)
