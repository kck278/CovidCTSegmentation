import os
import pprint
import torch
import torchmetrics
import numpy as np
import nibabel as nib
from typing import List, Tuple
from metrics import confusion_matrix, sensitivity, specificity, sorensen_dice, g_mean, precision, f2_score
        

def get_y(prediction_folder: str, ground_truth_folder: str):
    prediction_file_names, prediction_indices = get_prediction_names_indices(prediction_folder)
    gt_file_names = get_ground_truth_names(ground_truth_folder, prediction_indices)
    assert len(prediction_file_names) == len(gt_file_names)

    y_hat = images_to_tensor(prediction_file_names, prediction_folder)
    y = images_to_tensor(gt_file_names, ground_truth_folder)
    return y_hat, y


def get_prediction_names_indices(prediction_folder: str):
    prediction_file_names = []
    prediction_indices = []

    for file in os.listdir(prediction_folder):
        if file.endswith(".nii.gz"):
            prediction_file_names.append(file)
            prediction_indices.append(file.split(".")[0][-3:]+".nii.gz")

    return sorted(prediction_file_names), tuple(prediction_indices)


def get_ground_truth_names(ground_truth_folder: str, indices: Tuple[str]):
    gt_file_names = []

    for file in os.listdir(ground_truth_folder):
        if file.endswith(indices):
            gt_file_names.append(file)

    return sorted(gt_file_names)


def images_to_tensor(image_names: List[str], folder_name: str):
    all_images = []

    for file in image_names:
        nifti_path = os.path.join(folder_name, file)
        img = nib.load(nifti_path)
        img_arr = np.array(img.dataobj)[:, :, 0].tolist()
        all_images.append(img_arr)
        
    all_images = np.array(all_images)
    all_images = torch.from_numpy(all_images)
    return all_images


def calculate_metrics(
    y_hat: torch.FloatTensor, 
    y: torch.FloatTensor, 
    num_classes: int
) -> dict:
    '''Calculates different metrices for given batch.

    Args:
        y_hat: Tensor containing the predicted clases.
        y: Tensor containing the true labels.
        step: Description of the current step.
    Returns:
        Dictionary containing the metrices (accuracy, sensitivity, specificity, 
        sorensen dice, geometric mean, precision, f2-score).
    '''
    accuracy = torchmetrics.Accuracy().to(y_hat.device)

    metrics = { }
    tp_count = 0

    for i in range(1, num_classes):
        y_hat_copy = y_hat.detach().clone()
        y_copy = y.detach().clone()

        y_hat_copy[y_hat_copy != i] = 0
        y_copy[y_copy != i] = 0

        tp, tn, fp, fn = confusion_matrix(y_hat_copy, y_copy)
        tp_count += tp

        sens = sensitivity(tp, fn)
        spec = specificity(tn, fp)
        dice = sorensen_dice(tp, fp, fn)
        mean = g_mean(sens, spec)
        prec = precision(tp, fp)
        f2 = f2_score(prec, sens)

        metrics['sens_c' + str(i)] = sens
        metrics['spec_c' + str(i)] = spec
        metrics['dice_c' + str(i)] = dice
        metrics['mean_c' + str(i)] = mean
        metrics['prec_c' + str(i)] = prec
        metrics['f2_c' + str(i)] = f2
    
    metrics['acc'] = accuracy(y_hat, y.to(torch.int))
    return metrics

y_hat, y = get_y(
    prediction_folder="data/images/prediction/nnU-Net/binary_long",
    ground_truth_folder="data/images/prediction/nnU-Net/true_binary"
)

# Set num_classes (third parameter) either to 2 or 4 depending on binary / muli class prediction
metrics = calculate_metrics(y_hat, y, 2)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(metrics)