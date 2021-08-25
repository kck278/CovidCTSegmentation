import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert_nifi_to_png(nifti_path, png_path, binary=False):
    img = nib.load(nifti_path)

    img_arr = np.array(img.dataobj)
    print(img_arr.shape)
    print(img_arr[:, :, 0].shape)
    slices = img_arr.shape[-1]
    fill = len(str(slices))

    for i in tqdm(range(1, slices+1)):
        number = str(i).zfill(fill)
        image_name = png_path.format(number)
        image_save = np.rot90(img_arr[:, :, i-1], k=3)
        image_save = np.fliplr(image_save)
        if binary:
            image_save = np.clip(image_save, 0, 1)
        plt.imsave(fname=image_name, arr=image_save, cmap='gray', format='png')

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
convert_nifi_to_png('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_im.nii.gz',
                    'images/lung/lung_{}.png')
# mask conversion
convert_nifi_to_png('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_mask.nii.gz',
                    'images/mask/mask_{}.png')
# binary mask conversion
convert_nifi_to_png('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_mask.nii.gz',
                    'images/binary_mask/binary_mask_{}.png',
                    binary=True)
