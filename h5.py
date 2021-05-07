from config import args
import numpy as np
import h5py
from scipy.io import wavfile
import cv2 as cv
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

def get_files(datadir, dataset, fold):
    with open(datadir + "/" + dataset + ".txt", "r") as f:
        lines = f.readlines()
    datalist = [datadir + "/" + fold + "/" + line.strip() for line in lines]
    return datalist


def fill_in_data(datalist, wav, png, moco_model, device):
    for i, item in tqdm(enumerate(datalist)):
        sampFreq, inputAudio = wavfile.read(item + '.wav')
        wav[i] = np.array(inputAudio)
        vidInp = cv.imread(item + '.png') / 255
        vidInp = np.array(np.split(vidInp, range(112, len(vidInp[0]), 112), axis=1))
        vidInp = torch.tensor(vidInp).transpose(2, 3).transpose(1, 2).float().to(device)
        png[i] = moco_model(vidInp).cpu().numpy().flatten()


def main():
    torch.cuda.set_device(args["GPU_ID"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")

    datadir = args['DATA_DIRECTORY']
    pretrain_datalist = get_files(datadir, 'pretrain', 'pretrain')
    preval_datalist = get_files(datadir, 'preval', 'pretrain')
    train_datalist = get_files(datadir, 'train', 'main')
    val_datalist = get_files(datadir, 'val', 'main')
    test_datalist = get_files(datadir, 'test', 'main')

    f = h5py.File(args["HDF5_DIRECTORY"], "w")
    dt = h5py.vlen_dtype(np.dtype('int32'))
    pretrain_wav = f.create_dataset('pretrain_wav', (len(pretrain_datalist),), dtype=dt)
    preval_wav = f.create_dataset('preval_wav', (len(preval_datalist),), dtype=dt)
    train_wav = f.create_dataset('train_wav', (len(train_datalist),), dtype=dt)
    val_wav = f.create_dataset('val_wav', (len(val_datalist),), dtype=dt)
    test_wav = f.create_dataset('test_wav', (len(test_datalist),), dtype=dt)

    dt = h5py.vlen_dtype(np.dtype('float32'))
    pretrain_png = f.create_dataset('pretrain_png', (len(pretrain_datalist),), dtype=dt)
    preval_png = f.create_dataset('preval_png', (len(preval_datalist),), dtype=dt)
    train_png = f.create_dataset('train_png', (len(train_datalist),), dtype=dt)
    val_png = f.create_dataset('val_png', (len(val_datalist),), dtype=dt)
    test_png = f.create_dataset('test_png', (len(test_datalist),), dtype=dt)

    moco_model = models.__dict__['resnet50'](num_classes=2048)
    for name, param in moco_model.named_parameters():
        param.requires_grad = False
    checkpoint = torch.load(args['MOCO_DIRECTORY'], map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q.fc.0'):
            state_dict['fc' + k[len("module.encoder_q.fc.0"):]] = state_dict[k]
        elif k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = moco_model.load_state_dict(state_dict, strict=False)
    moco_model.to(device)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([
    #     transforms.RandomResizedCrop(112),
    #     transforms.RandomHorizontalFlip(),
    #     normalize,
    # ])

    # fill_in_data(pretrain_datalist, pretrain_wav, pretrain_png, moco_model, device)
    # fill_in_data(preval_datalist, preval_wav, preval_png, moco_model, device)
    fill_in_data(train_datalist, train_wav, train_png, moco_model, device)
    fill_in_data(val_datalist, val_wav, val_png, moco_model, device)
    # fill_in_data(test_datalist, test_wav, test_png, moco_model, device)
    f.close()


if __name__ == "__main__":
    main()
