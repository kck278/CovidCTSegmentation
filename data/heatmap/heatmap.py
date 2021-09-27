import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def heatmap_paper():
    img = nib.load('data/images/nifti/mask/tr_mask.nii.gz')
    img_arr = np.array(img.dataobj)
    print(img_arr.shape)

    summed = np.sum(img_arr, axis=2)
    print(summed.shape)

    plt.imshow(summed, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(
        'data/heatmap/heatmap_paper.png',
        bbox_inches='tight',
        pad_inches=0
    )


def heatmap_class_specific(class_number: float):
    img = nib.load('data/images/nifti/mask/tr_mask.nii.gz')
    img_arr = np.array(img.dataobj)
    img_arr = (img_arr == class_number).astype(int)
    print(img_arr.shape)

    summed = np.sum(img_arr, axis=2)
    print(summed.shape)

    plt.imshow(summed, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(
        'data/heatmap/heatmap_class_{}.png'.format(class_number),
        bbox_inches='tight',
        pad_inches=0
    )


def heatmap_infected_tissue():
    img = nib.load('data/images/nifti/mask/tr_mask.nii.gz')
    img_arr = np.array(img.dataobj)
    img_arr = (img_arr > 0).astype(int)
    print(img_arr.shape)

    summed = np.sum(img_arr, axis=2)
    print(summed.shape)

    plt.imshow(summed, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.savefig(
        'data/heatmap/heatmap_infected_tissue.png',
        bbox_inches='tight',
        pad_inches=0
    )


heatmap_paper()
heatmap_infected_tissue()
heatmap_class_specific(1.0)
heatmap_class_specific(2.0)
heatmap_class_specific(3.0)
