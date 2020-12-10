import os
import torch
import numpy as np
import argparse
import nibabel as nib
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Tensor stock.')
parser.add_argument('--reduction', type=int, default=1, help='Reduction of images.')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use.')
args = parser.parse_args()

print('>> Creates BraTs 3D dataset for reduction {} from preprocessed (normalized) BraTs.'.format(args.reduction))
os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
torch.set_num_threads(args.num_threads)

# ---------------------------------------------------------------
# PATHS
assert args.reduction in [0, 1, 2]
HOME_PATH = ...
data_nifti_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/2_t1ce_normalized')
data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/2_t1ce_normalized')
path_to_tensor_train = os.path.join(data_tensor_path, str(args.reduction) + '_reduction', 'train')
path_to_tensor_test = os.path.join(data_tensor_path, str(args.reduction) + '_reduction', 'test')

if not os.path.exists(data_nifti_path):
    raise AssertionError

if not os.path.exists(path_to_tensor_train):
    os.makedirs(path_to_tensor_train)
if not os.path.exists(path_to_tensor_test):
    os.makedirs(path_to_tensor_test)

# ---------------------------------------------------------------
# CONVERSION

fn_args = []
t_ = ('train', 'test')

for t in t_:
    path_to_nifti_t = os.path.join(data_nifti_path, t)
    path_to_tensor_t = os.path.join(data_tensor_path, str(args.reduction) + '_reduction', t)
    elts = [_[:-7] for _ in os.listdir(path_to_nifti_t) if _[-7:] == '.nii.gz']
    for elt in elts:
        fn_args.append((os.path.join(path_to_nifti_t, elt + '.nii.gz'), os.path.join(path_to_tensor_t, elt + '.pt')))


# ---------------------------------------------------------------
# SAVE GLOBAL AFFINE PARAMETER (assuming all affine parameters are identical)

random_nifti_reference, _ = fn_args[0]     # random nifti path | path_to_tensor_train
np_affine = nib.load(random_nifti_reference).affine
np.save(file=os.path.join(path_to_tensor_train, 'affine.npy'), arr=np_affine)
np.save(file=os.path.join(path_to_tensor_test, 'affine.npy'), arr=np_affine)
with np.printoptions(precision=3, suppress=True):
    print('>> Affine matrix saved :\n', np_affine)
print('>> Creating 3D Tensors ...')


def launch(l_args):
    import torch
    import nibabel as nib
    REDUCTION = 2 ** args.reduction

    path_in, path_out = l_args
    img = nib.load(path_in)
    img_affined = nib.Nifti1Image(img.get_data()[40:-40, 24:-24, 14:-13], img.affine)
    tensor_affined = torch.from_numpy(img_affined.get_data()).float()     # .to(DEVICE)
    tensor_red = torch.nn.functional.interpolate(tensor_affined.unsqueeze(0).unsqueeze(0),
                                                 scale_factor=1./REDUCTION,
                                                 mode='trilinear',
                                                 align_corners=False).squeeze(0) if REDUCTION > 1 \
        else tensor_affined.unsqueeze(0)
    torch.save(tensor_red.detach(), path_out)


# ---------------------------------------------------------------
# POOL OPERATION

with Pool(os.cpu_count()) as pool:
    pool.map(launch, fn_args)

print('>> Dataset built successfully.')

