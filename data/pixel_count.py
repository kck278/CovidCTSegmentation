import os
import numpy as np
import nibabel as nib
import pandas as pd


def calculate_statistics(binary: bool=False) -> pd.DataFrame:
    mask_dir = '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/mask'
    mask_names = sorted(os.listdir(mask_dir))

    pixel_dict = {}
    img_dict = {}

    for mask in mask_names:
        img_path = os.path.join(mask_dir, mask)
        img = nib.load(img_path)
        img_arr = np.array(img.dataobj)

        if binary:
            img_arr = np.clip(img_arr, 0, 1)

        unique, counts = np.unique(img_arr, return_counts=True)

        for i in range(len(unique)):
            if unique[i] in pixel_dict.keys():
                img_dict[unique[i]] += img_arr.size
                pixel_dict[unique[i]] += counts[i]
            else:
                pixel_dict[unique[i]] = counts[i]
                img_dict[unique[i]] = img_arr.size

    freqs = []

    for i in range(len(unique)):
        freq = pixel_dict[i] / img_dict[i]
        freqs.append(freq)

    med_freq = np.median(np.array(freqs))
    class_weigts = []

    for freq in freqs:
        class_weigt = med_freq / freq
        class_weigts.append(class_weigt)

    data = []

    for i in range(len(unique)):
        data.append(['C' + str(i), pixel_dict[i], img_dict[i], class_weigts[i]])

    return pd.DataFrame(data=np.array(data), columns=['Class', 'Pixel Count', 'Image Pixel count', 'Class Weight'])


def print_statistics():
    print('-----------------------------------------------------------')
    print('MULTILABEL')
    df_multilabel = calculate_statistics()
    print(df_multilabel)
    print('-----------------------------------------------------------')
    print('BINARY')
    df_binary = calculate_statistics(binary=True)
    print(df_binary)
    print('-----------------------------------------------------------')

# print_statistics()
