import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import tqdm

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.stats import entropy


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str,
                    help=('Path to the generated images'))
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size to use')
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def inception_score(data_path, gpu='', batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs

    data_path -- path to folder with images in class_dirs
    gpu -- id of GPU to be used (e.g. 0)
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """

    cuda = False if gpu == '' else True
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    imgs = dset.ImageFolder(root=data_path, transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    def get_pred(x):
        if resize:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).detach().cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        # import pdb; pdb.set_trace()
        batch = batch[0].type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in tqdm.tqdm(range(splits)):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':

    args = parser.parse_args()

    print(inception_score(args.path, gpu=args.gpu, batch_size=args.batch_size, resize=True, splits=10))

# python inception_score.py /path/to/dir/of/images --gpu 0
