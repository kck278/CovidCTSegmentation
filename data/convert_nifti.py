import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def convert_nifi_to_single_slice(nifti_path, new_path, binary=False):
    img = nib.load(nifti_path)

    #'C:/Users/sophi/Documents/0_Master/AML/CovidCTSegmentation/data/images/lung/lung_{}.png'
    img_arr = np.array(img.dataobj)
    print(img_arr.shape)
    print(img_arr[:, :, 0].shape)
    slices = img_arr.shape[-1]
    fill = len(str(slices))

    for i in tqdm(range(1, slices+1)):
        number = str(i).zfill(fill)
        image_name = new_path.format(number)
        image_save = img_arr[:, :, i-1]
        image_save = np.fliplr(image_save)
        if binary:
            image_save = np.clip(image_save, 0, 1)
        img_nifti = nib.Nifti1Image(image_save, img.affine, img.header)
        nib.save(img_nifti, image_name)


# specify the directory where you want to store images/ maksks/ binary masks
if not os.path.exists('images/lung'):
    os.makedirs('images/lung')
if not os.path.exists('images/mask'):
    os.makedirs('images/mask')
if not os.path.exists('images/binary_mask'):
    os.makedirs('images/binary_mask')

# specify the directory where nifty files are stored and where png files should be stored. png files are enumerated
# (therefore add {} in filename)

# image conversion
convert_nifi_to_single_slice('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_im.nii.gz',
                             'images/lung/lung_{}.nii.gz')
# mask conversion
convert_nifi_to_single_slice('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_mask.nii.gz',
                             'images/mask/mask_{}.nii.gz')
# binary mask conversion
convert_nifi_to_single_slice('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_mask.nii.gz',
                             'images/binary_mask/binary_mask_{}.nii.gz',
                             binary=True)
