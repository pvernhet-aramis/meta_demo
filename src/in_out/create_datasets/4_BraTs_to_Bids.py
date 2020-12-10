import os
import shutil
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Rearrange nifti original dataset to .')
parser.add_argument('--num_threads', type=int, default=36, help='Number of threads to use.')
args = parser.parse_args()

print('>> Creating BIDS structure from nifti files.')
os.environ['OMP_NUM_THREADS'] = str(args.num_threads)

# ---------------------------------------------------------------
# PATHS
HOME_PATH = ...
data_nifti_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/2_datasets/1_brats_2019')
data_segmented_path = os.path.join(HOME_PATH, 'Data/MICCAI_dataset/4_segmentations/BIDS')

if not os.path.exists(data_nifti_path):
    raise AssertionError

# ---------------------------------------------------------------
# Create BIDS directories accordingly
# BIDS_Dataset/
# ├── sub-CLNC01/
# │   │   ├── ses-M00/
# │   │   │   ├── anat/
# │   │   │   │   └── sub-CLNC01_ses-M00_T1w.nii.gz


def create_bids(t_):

    # MAIN DIRECTORIES
    path_to_nifti_t = os.path.join(data_nifti_path, t_)
    path_to_results_t = os.path.join(data_segmented_path, t_)
    os.makedirs(path_to_results_t, exist_ok=True)

    csv_list = []
    # DIRECTORIES : (HGG | LGG) - NAME - {.nii.gz, .nii.gz, ...}
    for root, _, files in tqdm(os.walk(path_to_nifti_t, topdown=True)):
        t1_filenames = [name for name in files if name.endswith('t1ce.nii.gz')]
        assert len(t1_filenames) <= 1, "Check directory, has more than one T1 file."

        if len(t1_filenames) == 1:
            # Safety checks
            t1_filename = t1_filenames[0]
            tumor_status, name = root.split('/')[-2:] if t_ == '1_training' else ('XGG', root.split('/')[-1])
            identification = tumor_status + name.replace('_', '')

            # BIDS structure
            BIDS_path = os.path.join(path_to_results_t, '-'.join(['sub', identification]), 'ses-M00', 'anat')
            BIDS_filename = '_'.join(['-'.join(['sub', identification]), 'ses-M00', 'T1w.nii.gz'])
            os.makedirs(BIDS_path, exist_ok=True)

            # Transfer data
            src_file, dest_file = os.path.join(root, t1_filename), os.path.join(BIDS_path, BIDS_filename)
            _ = shutil.copy(src_file, dest_file)
            csv_list.append({'src_file': src_file, 'dest_file': dest_file, 'tumor': tumor_status})

    # converts list of dicts to csv
    with open(os.path.join(path_to_results_t, 'table_links.csv'), 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, csv_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(csv_list)


create_bids('1_training')
create_bids('2_validation')

print('>> BIDS successfully created.')

