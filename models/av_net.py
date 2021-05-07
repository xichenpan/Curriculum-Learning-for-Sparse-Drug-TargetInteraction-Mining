import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from torch.nn.utils.rnn import pad_sequence
import math
from models.GraphModels import GraphNeuralNetwork

class ModalityNormalization(nn.Module):
    """
    batch*frame*features
    """

    def __init__(self):
        super(ModalityNormalization, self).__init__()

    def forward(self, inputBatch):
        meanBatch = torch.mean(inputBatch, dim=2, keepdim=True)
        varBatch = torch.std(inputBatch, dim=2, keepdim=True)
        return (inputBatch - meanBatch) / varBatch


class AVNet(nn.Module):
    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    # args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"]
    def __init__(self, task, wav2vecdir, dModel, nHeads, numLayers, peMaxLen, audinSize, vidinSize, fcHiddenSize,
                 dropout, numClasses, reqInpLen):
        super(AVNet, self).__init__()
        self.task = task
        self.reqInpLen = reqInpLen
        # pretrained model
        # wav2vec2.0
        wav2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vecdir])
        wav2vec_model = wav2vec_model[0]
        wav2vec_model.eval()
        for name, param in wav2vec_model.named_parameters():
            param.requires_grad = False
        self.wav2vec_model = wav2vec_model

        self.graph_model = GraphNeuralNetwork(
            in_dim=DTIdataset.drug_dataset.embedding_dim,
            out_dim=args.d_model,
            layer_type=args.graph_layer,
            num_pre=args.mlp_depth,
            num_graph_layer=args.graph_depth,
            head=args.GAT_head
        )



        # AVNET
        self.audioConv = nn.Conv1d(audinSize, dModel, kernel_size=2, stride=2, padding=0)
        self.videoConv = nn.Conv1d(vidinSize, dModel, kernel_size=1, stride=1, padding=0)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize,
                                                  dropout=dropout)
        self.ModalityNormalization = ModalityNormalization()
        self.jointConv = nn.Conv1d(2 * dModel, dModel, kernel_size=1, stride=1, padding=0)
        self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.videoEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.jointDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
        self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
        return

    def forward(self, inputBatch, reqInpLen, opmode):
        audioBatch, aud_mask, videoBatch, vid_len = inputBatch
        audioBatch, aud_mask = self.wav2vec_model(audioBatch, padding_mask=aud_mask, mask=False,
                                                  features_only=True).values()
        videoBatch = list(torch.split(videoBatch, vid_len.tolist(), dim=0))

        aud_len = torch.sum(~aud_mask, dim=1)
        dismatch = aud_len - 2 * vid_len
        vid_padding = torch.ceil(torch.div(dismatch, 2)).int()
        vid_padding = vid_padding * (vid_padding > 0)
        aud_padding = 2 * vid_padding - dismatch

        if self.task == "pretrain":
            mask = (vid_padding + vid_len) > reqInpLen
            vid_padding = mask * vid_padding + (~mask) * (reqInpLen - vid_len)
            mask = (aud_padding + aud_len) > 2 * reqInpLen
            aud_padding = mask * aud_padding + (~mask) * (2 * reqInpLen - aud_len)
        else:
            mask = (vid_padding + vid_len) > self.reqInpLen
            vid_padding = mask * vid_padding + (~mask) * (self.reqInpLen - vid_len)
            mask = (aud_padding + aud_len) > 2 * self.reqInpLen
            aud_padding = mask * aud_padding + (~mask) * (2 * self.reqInpLen - aud_len)

        vid_leftPadding = torch.floor(torch.div(vid_padding, 2)).int()
        vid_rightPadding = torch.ceil(torch.div(vid_padding, 2)).int()
        aud_leftPadding = torch.floor(torch.div(aud_padding, 2)).int()
        aud_rightPadding = torch.ceil(torch.div(aud_padding, 2)).int()

        audioBatch = audioBatch.unsqueeze(1).unsqueeze(1)
        audioBatch = list(audioBatch)
        for i, _ in enumerate(audioBatch):
            pad = nn.ReplicationPad2d(padding=(0, 0, aud_leftPadding[i], aud_rightPadding[i]))
            audioBatch[i] = pad(audioBatch[i][:, :, :aud_len[i]]).squeeze(0).squeeze(0)
            pad = nn.ReplicationPad2d(padding=(0, 0, vid_leftPadding[i], vid_rightPadding[i]))
            videoBatch[i] = pad(videoBatch[i].unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        audioBatch = pad_sequence(audioBatch, batch_first=True)
        videoBatch = pad_sequence(videoBatch, batch_first=True)
        vid_len = vid_len + vid_padding
        vid_len = vid_len.long()
        mask = torch.zeros(videoBatch.shape[:-1], device=videoBatch.device)  # [16, 99]
        mask[(torch.arange(mask.shape[0]), vid_len - 1)] = 1
        mask = (1 - mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        inputLenBatch = vid_len

        if opmode == "AO":
            videoBatch = None
        elif opmode == "VO":
            audioBatch = None

        if audioBatch is not None:
            audioBatch = audioBatch.transpose(1, 2)
            audioBatch = self.audioConv(audioBatch)
            # audioBatch = F.leaky_relu(audioBatch)
            audioBatch = audioBatch.transpose(1, 2).transpose(0, 1)
            audioBatch = self.positionalEncoding(audioBatch)
            audioBatch = self.audioEncoder(audioBatch, src_key_padding_mask=mask)
            audioBatch = self.ModalityNormalization(audioBatch)

        if videoBatch is not None:
            videoBatch = videoBatch.transpose(1, 2)
            videoBatch = self.videoConv(videoBatch)
            # videoBatch = F.leaky_relu(videoBatch)
            videoBatch = videoBatch.transpose(1, 2).transpose(0, 1)
            videoBatch = self.positionalEncoding(videoBatch)
            videoBatch = self.videoEncoder(videoBatch, src_key_padding_mask=mask)
            videoBatch = self.ModalityNormalization(videoBatch)

        if (audioBatch is not None) and (videoBatch is not None):
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
            jointBatch = self.jointConv(jointBatch)
            # jointBatch = F.leaky_relu(jointBatch)
            jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()

        # jointBatch = self.positionalEncoding(jointBatch)
        jointBatch = self.jointDecoder(jointBatch, src_key_padding_mask=mask)
        jointBatch = jointBatch.transpose(0, 1).transpose(1, 2)
        jointBatch = self.outputConv(jointBatch)
        jointBatch = jointBatch.transpose(1, 2).transpose(0, 1)
        # print(F.softmax(jointBatch,2))
        outputBatch = F.log_softmax(jointBatch, dim=2)
        return inputLenBatch, outputBatch
