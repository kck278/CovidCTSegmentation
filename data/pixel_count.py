import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import nibabel as nib
import pandas as pd

mask_dir = "/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/mask"
mask_names = sorted(os.listdir(mask_dir))
pixel_c0 = 0
pixel_c1 = 0
pixel_c2 = 0
pixel_c3 = 0

img_pixel_c0 = 0
img_pixel_c1 = 0
img_pixel_c2 = 0
img_pixel_c3 = 0
for mask in mask_names:
    img_path = os.path.join(mask_dir, mask)
    img = nib.load(img_path)
    img_np = np.array(img.dataobj)

    unique, counts = np.unique(img_np, return_counts=True)
    occurrences = dict(zip(unique, counts))
    if np.any(img_np == 0):
        pixel_c0 += occurrences[0.0]
        img_pixel_c0 += img_np.size
    if np.any(img_np == 1):
        pixel_c1 += occurrences[1.0]
        img_pixel_c1 += img_np.size
    if np.any(img_np == 2):
        pixel_c2 += occurrences[2.0]
        img_pixel_c2 += img_np.size
    if np.any(img_np == 3):
        pixel_c3 += occurrences[3.0]
        img_pixel_c3 += img_np.size

d = {'class': ['C0', 'C1', 'C2', 'C3'], 'Pixel count': [pixel_c0, pixel_c1, pixel_c2, pixel_c3],
     'Image pixel count': [img_pixel_c0, img_pixel_c1, img_pixel_c2, img_pixel_c3]}
df = pd.DataFrame(data=d)
print(df)
