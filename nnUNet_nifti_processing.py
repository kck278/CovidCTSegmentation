import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def convert_nifi_to_single_slice(nifti_path, new_path, test_path=None, binary=False):
    test_set = [79, 15, 77, 17, 43, 83, 22, 33, 61, 36, 41, 48, 18, 86, 69, 56, 59, 16]
    img = nib.load(nifti_path)

    img_arr = np.array(img.dataobj)
    print(img_arr.shape)
    print(img_arr[:, :, 0].shape)
    slices = img_arr.shape[-1]
    fill = len(str(slices))

    test = img_arr[:,:,0]
    test = np.expand_dims(test, 2)

    for i in tqdm(range(1, slices+1)):
        if i in test_set and test_path is None:
            continue

        number = str(i).zfill(fill)

        if i in test_set:
            image_name = test_path.format(number)
        else:
            image_name = new_path.format(number)

        image_save = img_arr[:, :, i-1]
        image_save = np.fliplr(image_save)

        if binary:
            image_save = np.clip(image_save, 0, 1)
            
        image_save = np.expand_dims(image_save, 2)
        img_nifti = nib.Nifti1Image(image_save, img.affine, img.header)
        nib.save(img_nifti, image_name)

# convert_nifi_to_single_slice("data/images/nifti/scan/tr_scan.nii.gz",
#  "nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/imagesTr/cov_{}_0000.nii.gz",
#  "nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/imagesTs/cov_{}_0000.nii.gz",
# False)

# convert_nifi_to_single_slice("data/images/nifti/mask/tr_mask.nii.gz",
#  "nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/labelsTr/cov_{}.nii.gz",
# binary=True)

# prepare dataset.json

# from nnunet.dataset_conversion.utils import generate_dataset_json

# generate_dataset_json("nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/dataset.json",
# "nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/imagesTr",
# "nnUNet/nnUNet_raw/nnUNet_raw_data/Task504_CovidBinary/imagesTs",
#  ('CT',), labels={0:'background', 1:'infected'}, dataset_name="COVID-19 CT segmentation dataset binary")