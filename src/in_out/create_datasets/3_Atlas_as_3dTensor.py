import os
import torch
import shutil
import nibabel as nib
from nilearn.image import resample_img
import itk

os.environ['OMP_NUM_THREADS'] = str(30)
torch.set_num_threads(30)

# ---------------------------------------------------------------
# PATHS
HOME_PATH = ...
atlas_name = ...
data_nifti_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/1_brats_2019/{}.nii'.format(atlas_name))
data_tensor_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/{}.pt'.format(atlas_name))
reference_image_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/2_t1ce_normalized/train/lgg_TCIA12_466_1_t1ce.nii.gz')

# i) ---- Resampling
img = nib.load(data_nifti_path)
target_img = nib.load(reference_image_path)
img = resample_img(img, target_affine=target_img.affine)

# ii) ---- Cropping
img_affined = nib.Nifti1Image(img.get_fdata()[40:-40, 24:-24, 14:-13], target_img.affine)
temp_save = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/{}_temp.nii.gz'.format(atlas_name))
nib.save(img_affined, temp_save)

# iii) ---- Histogram alignment
PixelType = itk.F
ImageType = itk.Image[PixelType, 3]
target_img_itk = itk.imread(reference_image_path, PixelType)
img_in = itk.imread(temp_save, PixelType)
img_out = itk.HistogramMatchingImageFilter(img_in, target_img_itk)
img_out = itk.RescaleIntensityImageFilter(img_out)
itk.imwrite(img_out, temp_save)    # saving again on previous temp position

# iv) ---- Tensor conversion
img_final = nib.load(temp_save)
tensor_final = torch.from_numpy(img_final.get_fdata()).squeeze().float()
torch.save(tensor_final.unsqueeze(0).detach(), data_tensor_path)
print('>> Atlas converted.')

try:
    shutil.rmtree(temp_save)
except OSError:
    os.remove(temp_save)
print('>> Temporary files deleted')

img_reloaded = torch.load(data_tensor_path).squeeze().detach().numpy()
nib.save(nib.Nifti1Image(img_reloaded, target_img.affine),
         os.path.join(HOME_PATH, 'Data/MICCAI_dataset/3_tensors3d/{}_reloaded.nii.gz'.format(atlas_name)))
print('>> Atlas reloaded version saved.')
