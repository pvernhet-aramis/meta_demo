import os
import torch
import numpy as np
import argparse
import itertools
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description='Spheres 3D tensor dataset.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use.')
args = parser.parse_args()

print('>> Creates centered Spheres 3D dataset.')
os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
torch.set_num_threads(args.num_threads)

# ---------------------------------------------------------------
# PATHS
HOME_PATH = ...
data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/2_spheres')
path_to_tensor_train = os.path.join(data_tensor_path, 'train')
path_to_tensor_test = os.path.join(data_tensor_path, 'test')

if not os.path.exists(path_to_tensor_train):
    os.makedirs(path_to_tensor_train)
if not os.path.exists(path_to_tensor_test):
    os.makedirs(path_to_tensor_test)

# ---------------------------------------------------------------
# GLOBAL HYPERPARAMETERS

img_size = 64
center = (img_size + 1.) / 2.0
minimal_radius = .1
sigma = 10.0**(1/3) / 2.   # sharper data than mock eyes dataset
tol = 1e-10
coordinates_x, coordinates_y, coordinates_z = np.meshgrid(np.arange(1, img_size + 1),
                                                          np.arange(1, img_size + 1),
                                                          np.arange(1, img_size + 1))


def generate_black_spheres(path, bor_min, bor_max, nb_pts_bor,
                           bir_min, bir_max, nb_pts_bir,
                           boc_min, boc_max, nb_pts_boc,
                           bic_min, bic_max, nb_pts_bic,
                           tc_min, tc_max,
                           nb_radii, nb_angles):

    bor_list = np.linspace(bor_min, bor_max, nb_pts_bor, endpoint=True)
    bir_list = np.linspace(bir_min, bir_max, nb_pts_bir, endpoint=True)
    boc_list = np.linspace(boc_min, boc_max, nb_pts_boc, endpoint=True)
    bic_list = np.linspace(bic_min, bic_max, nb_pts_bic, endpoint=True)

    i = 0
    for brain_outer_r, brain_inner_r, brain_outer_c, brain_inner_c in tqdm(itertools.product(bor_list, bir_list,
                                                                                             boc_list, bic_list)):
        for tumour_r, tumour_c in zip(np.random.uniform(min(minimal_radius, brain_inner_r), brain_outer_r, nb_radii),
                                      np.random.uniform(tc_min, tc_max, nb_radii)):
            for phi, costheta, u in zip(np.random.uniform(0, 2*np.pi, nb_angles),
                                        np.random.uniform(-1, 1, nb_angles),
                                        np.random.uniform(0, 1, nb_angles)):
                r = tumour_r * u ** (1./3)
                theta = np.arccos(costheta)
                tumor_center_x = r * np.sin(theta) * np.cos(phi)
                tumor_center_y = r * np.sin(theta) * np.sin(phi)
                tumor_center_z = r * np.cos(theta)

                path_out = os.path.join(path, 'sphere_{}.pt'.format(str(i)))
                img = np.zeros((img_size, img_size, img_size))
                img[((coordinates_x - center) ** 2) / (brain_outer_r * img_size) ** 2 +
                    ((coordinates_y - center) ** 2) / (brain_outer_r * img_size) ** 2 +
                    ((coordinates_z - center) ** 2) / (brain_outer_r * img_size) ** 2 <= 1.] = brain_outer_c
                img[((coordinates_x - center) ** 2) / (brain_inner_r * img_size) ** 2 +
                    ((coordinates_y - center) ** 2) / (brain_inner_r * img_size) ** 2 +
                    ((coordinates_z - center) ** 2) / (brain_inner_r * img_size) ** 2 <= 1.] = brain_inner_c

                # tumor circle | fixed color intensity
                img[((coordinates_x - tumor_center_x) ** 2) / (tumour_r * img_size) ** 2 +
                    ((coordinates_y - tumor_center_y) ** 2) / (tumour_r * img_size) ** 2 +
                    ((coordinates_z - tumor_center_z) ** 2) / (tumour_r * img_size) ** 2 <= 1.] = .8

                # Smoothing, clipping and saving as torch tensor
                img = gaussian_filter(img, sigma * img_size / 100.)  # smoothing of data
                img = (np.clip(img, tol, 1.0 - tol) * 255).astype('uint8')
                torch_img = torch.from_numpy(img).float().unsqueeze(0)    # add first channel dimension
                torch.save(torch_img, path_out)

                i += 1

# ---------------------------------------------------------------
# GENERATE DATASETS


generate_black_spheres(path_to_tensor_train, .35, .45, 5, .15, .25, 5, .1, .25, 5, .30, .60, 5, .75, 0.9,
                       nb_radii=5, nb_angles=10)
generate_black_spheres(path_to_tensor_test, .3, .5, 3, .1, .3, 3, .05, .25, 3, .25, .65, 5, .7, 0.95,
                       nb_radii=5, nb_angles=10)


print('>> Dataset built successfully.')

