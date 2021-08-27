import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

img = nib.load("C:/Users/sophi/Documents/0_Master/AML/CovidCTSegmentation/data/images/tr_im.nii.gz")

img_arr = np.array(img.dataobj)
print(img_arr.shape)
print(img_arr[:, :, 0].shape)
slices = img_arr.shape[-1]
fill = len(str(slices))

for i in tqdm(range(1, slices+1)):
    number = str(i).zfill(fill)
    image_name = 'C:/Users/sophi/Documents/0_Master/AML/CovidCTSegmentation/data/images/lung/lung_{}.png'.format(number)
    image_save = np.rot90(img_arr[:, :, i-1], k=3)
    image_save = np.fliplr(image_save)
    image_save = np.clip(image_save, 0, 1)
    plt.imsave(fname=image_name, arr=image_save, cmap='gray', format='png')
