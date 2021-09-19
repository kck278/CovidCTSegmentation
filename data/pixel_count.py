import os
import numpy as np
import pandas as pd
from PIL import Image


def calculate_statistics(num_classes: int=2, extended: bool=False) -> pd.DataFrame:
    mask_dir = '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/multilabel/512'
    mask_names = []

    if extended:
        for root, _, files in os.walk(mask_dir):
            for name in files:
                mask_name = os.path.join(root, name)
                mask_name = os.path.relpath(mask_name, mask_dir)
                mask_names.append(mask_name)
    else:
        mask_names = [f for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))]
    
    mask_names = sorted(mask_names)

    pixel_dict = {}
    img_dict = {}

    for mask in mask_names:
        img_path = os.path.join(mask_dir, mask)
        img_arr = np.asarray(Image.open(img_path).convert('L'))
        img_arr[img_arr == 85] = 1
        img_arr[img_arr == 170] = 2
        img_arr[img_arr == 255] = 3

        if num_classes == 2:
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

    for i in range(len(pixel_dict)):
        freq = pixel_dict[i] / img_dict[i]
        freqs.append(freq)

    med_freq = np.median(np.array(freqs))
    class_weigts = []

    for freq in freqs:
        class_weigt = med_freq / freq
        class_weigts.append(class_weigt)

    data = []

    for i in range(len(pixel_dict)):
        data.append(['C' + str(i), pixel_dict[i], img_dict[i], class_weigts[i]])

    return pd.DataFrame(data=np.array(data), columns=['Class', 'Pixel Count', 'Image Pixel count', 'Class Weight'])


def print_statistics():
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print('BINARY')
    df_binary = calculate_statistics(num_classes=2, extended=False)
    print(df_binary)
    print('-----------------------------------------------------------')
    print('MULTILABEL')
    df_multilabel = calculate_statistics(num_classes=4, extended=False)
    print(df_multilabel)
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print('BINARY - EXTENDED')
    df_binary = calculate_statistics(num_classes=2, extended=True)
    print(df_binary)
    print('-----------------------------------------------------------')
    print('MULTILABEL - EXTENDED')
    df_multilabel = calculate_statistics(num_classes=4, extended=True)
    print(df_multilabel)
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')


# print_statistics()
