### Base ###
import fnmatch
import os
import math

### Core ###
import numpy as np
import torch
from tqdm import tqdm

### IMPORTS ###
from src.in_out.data_miccai import *
from torchvision import datasets, transforms
from scipy.ndimage import zoom
import nibabel as nib
import PIL.Image as pimg
from torch.utils.data import Dataset, DataLoader
from src.support.base_miccai import gpu_numpy_detach

def resize_32_32(image):
    a, b = image.shape
    return zoom(image, zoom=(32 * 1. / a, 32 * 1. / b))


# -----------------------------------------------------------
# 2D Datasets
# -----------------------------------------------------------


def load_brats(data_path, number_of_images_train, number_of_images_test, random_seed=None):

    path_to_train = os.path.join(data_path, 'train')
    path_to_test = os.path.join(data_path, 'test')

    # Train
    intensities_train = []

    elts = sorted([_ for _ in os.listdir(path_to_train) if _[-3:] == 'png'], key=(lambda x: int(x.split('_')[0])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_train, elt)))
        intensities_train.append(torch.from_numpy(img).float())
    intensities_train = torch.stack(intensities_train)

    assert number_of_images_train <= intensities_train.size(0), \
        'Too many required train files. A maximum of %d are available' % intensities_train.size(0).shape[0]
    intensities_train = intensities_train[:number_of_images_train]

    # Test
    intensities_test = []

    elts = sorted([_ for _ in os.listdir(path_to_test) if _[-3:] == 'png'], key=(lambda x: int(x.split('_')[0])))
    for elt in elts:
        img = np.array(pimg.open(os.path.join(path_to_test, elt)))
        intensities_test.append(torch.from_numpy(img).float())
    intensities_test = torch.stack(intensities_test)

    assert number_of_images_test <= intensities_test.size(0), \
        'Too many required test files. A maximum of %d are available' % intensities_test.size(0).shape[0]
    intensities_test = intensities_test[:number_of_images_test]

    # Finalize
    intensities_template = torch.mean(intensities_train, dim=0)

    intensities_mean = float(torch.mean(intensities_train).detach().cpu().numpy())
    intensities_std = float(torch.std(intensities_train).detach().cpu().numpy())
    intensities_train = ((intensities_train - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_test = ((intensities_test - intensities_mean) / intensities_std).unsqueeze(1)
    intensities_template = ((intensities_template - intensities_mean) / intensities_std).unsqueeze(0)

    return intensities_train, intensities_test, intensities_template, intensities_mean, intensities_std

# -----------------------------------------------------------
# custom 2D DataLoader (avoids memory overload)
# -----------------------------------------------------------


class BilinearInterpolation:
    """Bilinear interpolation for reduction if reduction size is not one"""

    def __init__(self, reduction):
        self.red = reduction

    def __call__(self, x):
        """
        :param x: image (batch, channel, width, height)
        :return: interpolated image (dimension reduction)
        """
        assert len(x.size()) == 4, "only accepts inputs of dimensions (batch, channel, width, height)"
        reduced_tensor = torch.nn.functional.interpolate(x,
                                                         scale_factor=1. / self.red,
                                                         mode='bilinear', align_corners=False) if self.red > 1 else x
        return reduced_tensor


class TrilinearInterpolation:
    """Trilinear interpolation for reduction if reduction size is not one"""

    def __init__(self, reduction):
        self.red = reduction

    def __call__(self, x):
        """
        :param x: image (batch, channel, width, height, depth)
        :return: interpolated image (dimension reduction)

        NB: trick by adding batch dimension and removing it afterward
        """
        assert len(x.size()) == 5, "only accepts inputs of dimensions (batch, channel, width, height, depth)"
        reduced_tensor = torch.nn.functional.interpolate(x,
                                                         scale_factor=1. / self.red,
                                                         mode='trilinear', align_corners=False) if self.red > 1 else x
        return reduced_tensor


class ZeroOneT12DDataset(Dataset):
    """(Sub) Dataset of BraTs 3D already gathered into folder.
    downsampling | sliced at half of chosen dimension
    Rescaling of data to [0, 1] (via / 255)
    """

    def __init__(self, img_dir, nb_files, sliced_dim, reduction=1, init_seed=123, check_endswith='pt',
                 eps=1e-6, is_half=False):
        """
        Args:
            img_dir (string): Input directory - must contain all torch Tensors.
            nb_files (int): number of subset data to randomly select.
            data_file (string): File name of the train/test split file.
            init_seed (int): initialization seed for random data selection.
            check_endswith (string, optional): check for files extension.
        """
        assert len(check_endswith), "must check for valid files extension"
        self.img_dir = img_dir
        self.sliced_dim = sliced_dim
        self.reduction = reduction
        self.eps = eps
        self.base_transform = BilinearInterpolation(self.reduction)  # interpolation after slicing for better accuracy
        self.transform = transforms.Compose([
            self.base_transform,
            transforms.Lambda(lambda x: (x.div(255))),
            transforms.Lambda(lambda x: (torch.clamp(x, self.eps, 1. - self.eps)))
        ])
        self.is_half = is_half
        r = np.random.RandomState(init_seed)

        # Check path exists, and set nb_files to min if necessary
        if os.path.isdir(img_dir):
            candidates_tensors = [_ for _ in os.listdir(img_dir) if _.endswith(check_endswith)]
            nb_candidates = len(candidates_tensors)
            if nb_candidates < nb_files:
                print('>> Number of asked files {} exceeds number of available files {}'.format(nb_files, nb_candidates))
                print('>> Setting number of data to maximum available : {}'.format(nb_candidates))
                self.nb_files = nb_candidates
            else:
                print('>> Creating dataset with {} files (from {} available)'.format(nb_files, nb_candidates))
                self.nb_files = nb_files

            self.database = list(r.choice(candidates_tensors, size=self.nb_files, replace=False))
        else:
            raise Exception('The argument img_dir is not a valid directory.')

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        filename = self.database[idx]
        image_path = os.path.join(self.img_dir, filename)
        sample = torch.load(image_path).float()
        assert len(sample.size()) == 4, "expecting tensors saved as (channel, width, height, depth)"
        if self.sliced_dim == 0:
            sample = sample[:, sample.size(1 + self.sliced_dim) // 2]
        elif self.sliced_dim == 1:
            sample = sample[:, :, sample.size(1 + self.sliced_dim) // 2]
        elif self.sliced_dim == 2:
            sample = sample[:, :, :, sample.size(1 + self.sliced_dim) // 2]
        else:
            raise AssertionError
        sample = self.transform(sample.unsqueeze(0)).squeeze(0)  # (channel, width, height) with channel = 1
        if self.is_half:
            sample = sample.half()
        return sample, filename.split('.')[0]

    def compute_statistics(self):
        """
        Computes statistics in an online fashion (using Welford’s method)
        """
        print('>> Computing online statistics for dataset ...')
        for elt in tqdm(range(self.nb_files)):
            sample, _ = self.__getitem__(elt)
            image = sample.detach().clone()
            if elt == 0:
                current_mean = image
                current_var = torch.zeros_like(image)
            else:
                old_mean = current_mean.detach().clone()
                current_mean = old_mean + 1. / (1. + elt) * (image - old_mean)
                current_var = float(elt - 1) / float(elt) * current_var + 1. / (1. + elt) * (image - current_mean) * (image - old_mean)

        mean = current_mean.detach().clone().float()
        std = torch.clamp(torch.sqrt(current_var), self.eps).detach().clone().float()
        # std[np.where(std.cpu() <= self.eps)] = self.eps
        return mean, std


