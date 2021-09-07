import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image

def convert_nifi_to_single_slice(nifti_path, new_path, binary=False):
    img = nib.load(nifti_path)
    img_arr = np.array(img.dataobj)
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


def crop_lungmask(png_lung_dir, lung_mask_dir, new_path):
    # find all images
    lung_names = sorted(os.listdir(png_lung_dir))
    lung_mask_names = sorted(os.listdir(lung_mask_dir))
    assert len(lung_names) == len(lung_mask_names)
    fill = len(str(len(lung_names)))

    for i, (lung, mask) in tqdm(enumerate(zip(lung_names, lung_mask_names))):
        lung_img = sitk.ReadImage(os.path.join(png_lung_dir, lung))
        mask_img = sitk.ReadImage(os.path.join(lung_mask_dir, mask))

        lung_arr = sitk.GetArrayFromImage(lung_img)
        mask_arr = sitk.GetArrayFromImage(mask_img)

        lung_arr[mask_arr==0] = 0

        number = str(i+1).zfill(fill)
        save_itk_png(lung_arr, new_path + "/lung_{}.png".format(number))

def mask_to_png(mask_path, new_path, binary=False):
    img = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(img)

    if binary:
        new_path += "/binary/mask_{}.png"
    else:
        new_path += "/multilabel/mask_{}.png"

    slices = img_arr.shape[0]
    fill = len(str(slices))

    for i in tqdm(range(0, slices)):
        number = str(i + 1).zfill(fill)
        image_path = new_path.format(number)
        image_save = img_arr[i, :, :]

        if binary:
            image_save = np.clip(image_save, 0, 1)
            image_save *= 255
        else:
            image_save *= 85 

        save_pil_png(image_save, image_path)

def save_itk_png(img_array, save_path):
    img = sitk.GetImageFromArray(img_array)
    sitk.WriteImage(img, save_path)

    img = Image.open(save_path)
    img.thumbnail((256, 256), Image.ANTIALIAS)
    img.save(save_path)

def save_pil_png(img_array, save_path):
    img = Image.fromarray(img_array)
    img = img.convert('RGB')
    img.thumbnail((256, 256), Image.ANTIALIAS)
    img.save(save_path)

# image conversion
# crop_lungmask(
#     "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/scan",
#     "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung_mask",
#     "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung"
# )

# mask conversion
mask_to_png(
     "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_mask.nii.gz",
     "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask",
    True
)
