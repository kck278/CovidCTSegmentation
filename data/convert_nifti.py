import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

img = nib.load("/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/tr_im.nii.gz")

img_arr = np.array(img.dataobj)
print(img_arr.shape)
print(img_arr[:, :, 0].shape)
slices = img_arr.shape[-1]
fill = len(str(slices))

for i in tqdm(range(1, slices+1)):
    number = str(i).zfill(fill)
    image_name = 'images/lung/lung_{}.png'.format(number)
    image_save = np.rot90(img_arr[:, :, i-1], k=3)
    image_save = np.fliplr(image_save)
    plt.imsave(fname=image_name, arr=image_save, cmap='gray', format='png')
