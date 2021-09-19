import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm
from PIL import Image
from util import sorted_alphanumeric


def niftis_to_png(input: str, resolution: int=512, binary: bool=True):
    base_dir = '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images'
    nifti_path = os.path.join(base_dir, 'nifti', input, 'tr_' + input + '.nii.gz')
    extended_nifti_dir = os.path.join(base_dir, 'nifti', input, 'extended')
    save_dir = os.path.join(base_dir, 'png', input)
    is_mask = input == 'mask'

    if is_mask:
        if binary:
            save_dir = os.path.join(save_dir, 'binary', str(resolution))
        else:
            save_dir = os.path.join(save_dir, 'multilabel', str(resolution))
    
    nifti_to_png(input=input, nifti_path=nifti_path, save_dir=save_dir, binary=binary, resolution=resolution)
    extended_nifti_names = sorted_alphanumeric(os.listdir(extended_nifti_dir))

    for index, extended_name in enumerate(extended_nifti_names):
        extended_path = os.path.join(extended_nifti_dir, extended_name)
        extended_save_dir = os.path.join(save_dir, 'extended', str(index + 1))

        nifti_to_png(
            input=input,
            nifti_path=extended_path, 
            save_dir=extended_save_dir,
            binary=binary,
            resolution=resolution
        )


def nifti_to_png(input: str, nifti_path: str, save_dir: str, binary: bool, resolution: int):
    is_mask = input == 'mask'

    img = nib.load(nifti_path)
    img_arr = np.array(img.dataobj)

    slices = img_arr.shape[-1]
    fill = len(str(slices))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in tqdm(range(0, slices)):
        image_save = img_arr[:, :, i - 1]

        if is_mask:
            if binary:
                image_save = np.clip(image_save, 0, 1)
                image_save *= 255
            else:
                image_save *= 85 

        number = str(i + 1).zfill(fill)
        file_name = input + '_{}.png'
        save_path = os.path.join(save_dir, file_name.format(number))

        if is_mask:
            img = Image.fromarray(image_save)
            img = img.convert('RGB')
            img.thumbnail((resolution, resolution), Image.NEAREST)
            img.save(save_path)
        else:
            plt.imsave(fname=save_path, arr=image_save, cmap='gray', format='png')


def crop_lungmask(scan_dir: str, lung_mask_dir: str, save_dir: str, resolution: int, extended: bool=False):
    lung_names = sorted_alphanumeric(os.listdir(scan_dir))
    lung_mask_names = sorted_alphanumeric(os.listdir(lung_mask_dir))
    assert len(lung_names) == len(lung_mask_names)
    fill = len(str(len(lung_names)))

    if not extended:
        save_dir = os.path.join(save_dir, str(resolution))

    for i, (lung_name, mask_name) in tqdm(enumerate(zip(lung_names, lung_mask_names))):
        if lung_name == "extended":
            extended_scan_dir = os.path.join(scan_dir, 'extended')
            extended_lung_mask_dir = os.path.join(lung_mask_dir, 'extended')
            extended_scan_dir_names = sorted_alphanumeric(os.listdir(extended_scan_dir))
            extended_lung_mask_dir_names = sorted_alphanumeric(os.listdir(extended_lung_mask_dir))

            for (scan_dir_name, lung_mask_dir_name) in zip(extended_scan_dir_names, extended_lung_mask_dir_names):
                crop_lungmask(
                    scan_dir=os.path.join(extended_scan_dir, scan_dir_name), 
                    lung_mask_dir=os.path.join(extended_lung_mask_dir, lung_mask_dir_name),
                    save_dir=os.path.join(save_dir, 'extended', scan_dir_name),
                    resolution=resolution,
                    extended=True
                )
            continue
        
        lung_img = Image.open(os.path.join(scan_dir, lung_name))
        lung_arr = np.asarray(lung_img)

        mask_img = Image.open(os.path.join(lung_mask_dir, mask_name))
        mask_arr = np.asarray(mask_img)

        lung_arr[mask_arr == 0] = 0

        number = str(i + 1 if extended else i).zfill(fill)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, 'lung_{}.png'.format(number))

        img = Image.fromarray(lung_arr)
        img = img.convert('RGB')
        img.thumbnail((resolution, resolution), Image.ANTIALIAS)
        img.save(save_path)


# Nifti to PNG
niftis_to_png(input='scan', resolution=512)
niftis_to_png(input='lung_mask', resolution=512)
niftis_to_png(input='mask', binary=True, resolution=256)
niftis_to_png(input='mask', binary=True, resolution=512)
niftis_to_png(input='mask', binary=False, resolution=256)
niftis_to_png(input='mask', binary=False, resolution=512)

# Image conversion
crop_lungmask(
    scan_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/scan',
    lung_mask_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung_mask',
    save_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung',
    resolution=256
)

crop_lungmask(
    scan_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/scan',
    lung_mask_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung_mask',
    save_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung',
    resolution=512
)
