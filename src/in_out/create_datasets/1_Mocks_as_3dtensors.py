import os
import torch
import numpy as np
import argparse
import itertools
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Mock tensor dataset.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use.')
args = parser.parse_args()

print('>> Creates centered Mock Eyes 3D dataset.')
os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
torch.set_num_threads(args.num_threads)

# ---------------------------------------------------------------
# PATHS
HOME_PATH = '/network/lustre/dtlake01/aramis/users/paul.vernhet'
data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/1_eyes')
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
sigma = 10.0**(1/3)
tol = 1e-10
brain_r = 60
tumour_outer_r = 36
tumour_inner_r = 18.
coordinates_x, coordinates_y, coordinates_z = np.meshgrid(np.arange(1, img_size + 1),
                                                          np.arange(1, img_size + 1),
                                                          np.arange(1, img_size + 1))


def generate_black_eyes(path, x_min, x_max, y_min, y_max, z_min, z_max, oc_min, oc_max, ic_min, ic_max, nb_pts):

    dx_list = np.linspace(x_min, x_max, nb_pts, endpoint=True)
    dy_list = np.linspace(y_min, y_max, nb_pts, endpoint=True)
    dz_list = np.linspace(z_min, z_max, nb_pts, endpoint=True)
    oc_list = np.linspace(oc_min, oc_max, nb_pts, endpoint=True)
    ic_list = np.linspace(ic_min, ic_max, nb_pts, endpoint=True)

    for i, (dx, dy, dz, oc, ic) in tqdm(enumerate(itertools.product(dx_list, dy_list, dz_list, oc_list, ic_list))):
        path_out = os.path.join(path, 'black_eye_{}.pt'.format(str(i)))
        tumour_outer_c = oc
        tumour_inner_c = ic
        img = np.zeros((img_size, img_size, img_size))

        # Now replacing information of data to newer shade of gray matter
        img[((coordinates_x - center) ** 2) / (brain_r / dx * img_size / 100.) ** 2 +
            ((coordinates_y - center) ** 2) / (brain_r / dy * img_size / 100.) ** 2 +
            ((coordinates_z - center) ** 2) / (brain_r / dz * img_size / 100.) ** 2 <= 1.] = 0.5
        img[((coordinates_x - center) ** 2) / (tumour_outer_r / dx * img_size / 100.) ** 2 +
            ((coordinates_y - center) ** 2) / (tumour_outer_r / dy * img_size / 100.) ** 2 +
            ((coordinates_z - center) ** 2) / (
                        tumour_outer_r / dz * img_size / 100.) ** 2 <= 1.] = tumour_outer_c
        img[((coordinates_x - center) ** 2) / (tumour_inner_r / dx * img_size / 100.) ** 2 +
            ((coordinates_y - center) ** 2) / (tumour_inner_r / dy * img_size / 100.) ** 2 +
            ((coordinates_z - center) ** 2) / (
                        tumour_inner_r / dz * img_size / 100.) ** 2 <= 1.] = tumour_inner_c

        # Smoothing, clipping and saving as torch tensor
        img = gaussian_filter(img, sigma * img_size / 100.)  # smoothing of data
        img = (np.clip(img, tol, 1.0 - tol) * 255).astype('uint8')
        torch_img = torch.from_numpy(img).float().unsqueeze(0)    # add first channel dimension
        torch.save(torch_img, path_out)

# ---------------------------------------------------------------
# GENERATE DATASETS


generate_black_eyes(path_to_tensor_train, 1.55, 2.45, 1.55, 2.45, 1.55, 2.45, 0.05, 0.95, 0.05, 0.95, nb_pts=8)
generate_black_eyes(path_to_tensor_test, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 0., 1., 0., 1., nb_pts=7)

# train_dx_list = np.linspace(1.55, 2.45, 8, endpoint=True)
# train_dy_list = np.linspace(1.55, 2.45, 8, endpoint=True)
# train_dz_list = np.linspace(1.55, 2.45, 8, endpoint=True)
# train_oc_list = np.linspace(0.05, 0.95, 8, endpoint=True)
# train_ic_list = np.linspace(0.05, 0.95, 8, endpoint=True)
# train_path_out = [os.path.join(path_to_tensor_train, 'black_eye_{}.pt'.format(str(i)))
#                   for i in range(len(list(itertools.product(train_dx_list,
#                                                             train_dy_list,
#                                                             train_dz_list,
#                                                             train_oc_list,
#                                                             train_ic_list))))
#                   ]
#
# test_dx_list = np.linspace(1.5, 2.5, 7, endpoint=True)
# test_dy_list = np.linspace(1.5, 2.5, 7, endpoint=True)
# test_dz_list = np.linspace(1.5, 2.5, 7, endpoint=True)
# test_oc_list = np.linspace(0., 1., 7, endpoint=True)
# test_ic_list = np.linspace(0., 1., 7, endpoint=True)
# test_path_out = [os.path.join(path_to_tensor_test, 'black_eye_{}.pt'.format(str(i)))
#                  for i in range(len(list(itertools.product(test_dx_list,
#                                                            test_dy_list,
#                                                            test_dz_list,
#                                                            test_oc_list,
#                                                            test_ic_list))))
#                  ]
#
# f_args_grouped = list(zip(train_path_out, itertools.product(train_dx_list, train_dy_list, train_dz_list,
#                                                             train_oc_list, train_ic_list))) + \
#                  list(zip(test_path_out, itertools.product(test_dx_list, test_dy_list, test_dz_list,
#                                                             test_oc_list, test_ic_list)))
#
#
# def pooled_generate_black_eyes(fargs):
#     path_out, (dx, dy, dz, oc, ic) = fargs
#     tumour_outer_c = oc
#     tumour_inner_c = ic
#     img = np.zeros((img_size, img_size, img_size))
#
#     # Now replacing information of data to newer shade of gray matter
#     img[((coordinates_x - center) ** 2) / (brain_r / dx * img_size / 100.) ** 2 +
#         ((coordinates_y - center) ** 2) / (brain_r / dy * img_size / 100.) ** 2 +
#         ((coordinates_z - center) ** 2) / (brain_r / dz * img_size / 100.) ** 2 <= 1.] = 0.5
#     img[((coordinates_x - 0.5 * (1 + img_size)) ** 2) / (tumour_outer_r / dx * img_size / 100.) ** 2 +
#         ((coordinates_y - 0.5 * (1 + img_size)) ** 2) / (tumour_outer_r / dy * img_size / 100.) ** 2 +
#         ((coordinates_z - 0.5 * (1 + img_size)) ** 2) / (
#                     tumour_outer_r / dz * img_size / 100.) ** 2 <= 1.] = tumour_outer_c
#     img[((coordinates_x - 0.5 * (1 + img_size)) ** 2) / (tumour_inner_r / dx * img_size / 100.) ** 2 +
#         ((coordinates_y - 0.5 * (1 + img_size)) ** 2) / (tumour_inner_r / dy * img_size / 100.) ** 2 +
#         ((coordinates_z - 0.5 * (1 + img_size)) ** 2) / (
#                     tumour_inner_r / dz * img_size / 100.) ** 2 <= 1.] = tumour_inner_c
#
#     # Smoothing, clipping and saving as torch tensor
#     img = gaussian_filter(img, sigma * img_size / 100.)  # smoothing of data
#     img = (np.clip(img, tol, 1.0 - tol) * 255).astype('uint8')
#     torch_img = torch.from_numpy(img).float().unsqueeze(0)    # add first channel dimension
#     torch.save(torch_img, path_out)
#
#
# # ---------------------------------------------------------------
# # POOL OPERATION
#
# with Pool(os.cpu_count()) as pool:
#     pool.map(pooled_generate_black_eyes, f_args_grouped)


print('>> Dataset built successfully.')

