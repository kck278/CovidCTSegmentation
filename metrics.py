from typing import Optional
import torch
import numpy as np
from torch import nn
from data.pixel_count import calculate_statistics


def calculate_metrics(y_hat: torch.FloatTensor, y, weights: torch.FloatTensor, step: str):
    '''Calculates different metrices for given batch.

    Args:
        y_hat: Tensor containing the predicted class probabilities.
        y: Tensor containing the true labels.
        step: Description of the current step.
    Returns:
        Dictionary containing the metrices (accuracy, sensitivity, specificity, 
        sorensen dice, geometric mean, precision, f2-score).
    '''
    y_pred = y_hat.argmax(dim=1)
    tp, tn, fp, fn = confusion_matrix(y_pred, y)
    loss = cross_entropy_loss(y_hat, y, weights)

    acc = accuracy(tp, tn, fp, fn)
    sens = sensitivity(tp, fn)
    spec = specificity(tn, fp)
    dice = sorensen_dice(tp, fp, fn)
    mean = g_mean(sens, spec)
    prec = precision(tp, fp)
    f2 = f2_score(prec, sens)

    return {
        step + '_loss': loss, 
        step + '_acc': acc, 
        step + '_sens': sens, 
        step + '_spec': spec, 
        step + '_dice': dice,
        step +  '_mean': mean, 
        step + '_f2': f2
    }


def cross_entropy_loss(y_hat: torch.LongTensor, y: torch.FloatTensor, weight: Optional[torch.FloatTensor]):
    '''Calculates the difference between two probability distributions.
    Lower values indicate better performance.

    Args:
        y_hat: Tensor containing the predicted class probabilities.
        y: Tensor containing the true labels.
        weight: Optional tensor of class weights.
    Returns:
        Cross entropy loss.
    '''
    cross_entropy = nn.CrossEntropyLoss(weight=weight)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return cross_entropy(y_hat, y.type(torch.LongTensor).to(device=device))


def confusion_matrix(y_pred: torch.LongTensor, y: torch.FloatTensor):
    '''Calculates confusion matrix for further metric evaluation

    Args:
        y_hat: Tensor containing the predicted class probabilities.
        y: Tensor containing the true labels.
    Returns:
        Tuple containing the number of true positives, number of true negatives, 
        number of false positives and number of false negatives.
    '''
    conf_vec = y_pred / y
    tp = torch.sum(conf_vec == 1).item()
    tn = torch.sum(torch.isnan(conf_vec)).item()
    fp = torch.sum(conf_vec == float('inf')).item()
    fn = torch.sum(conf_vec == 0).item()
    return tp, tn, fp, fn


def accuracy(tp: int, tn: int, fp: int, fn: int):
    '''Calculates the fraction of correct predictions.

    Args:
        tp: Number of true positives.
        tn: Number of true negatives.
        fp: Number of false positives.
        fn: Number of false negatives.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    return (tp + tn) / (tp + tn + fp + fn)


def sensitivity(tp: int, fn: int):
    '''Calculates the probability of a infected labelled pixel given that it is really infected.

    Args:
        tp: Number of true positives.
        fn: Number of false negatives.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def specificity(tn: int, fp: int):    
    '''Calculates the probability of a healthy labelled pixel given that it is really healthy.

    Args:
        tn: Number of true negatives.
        fp: Number of false positives.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    if tn + fp == 0:
        return 0
    return tn / (tn + fp)


def sorensen_dice(tp: int, fp: int, fn: int):
    '''Calculates the similiarity between two samples (prediction and ground truth).

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    if 2 * tp + fp + fn == 0:
        return 0
    return (2 * tp) / (2 * tp + fp + fn)


def g_mean(sens: float, spec: float):
    '''Calculates the geometric average of sensitivity and specificity.

    Args:
        sens: Sensitivity.
        spec: Specificity.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    return np.sqrt(sens * spec)


def precision(tp: int, fp: int):
    '''Calculates the roportion of positive identifications that were actually correct.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def f2_score(prec: float, sens: float):
    '''Calculates the weighted mixture of precision and recall.

    Args:
        prec: Precision.
        sens: Sensitivity.
    Returns:
        Value between 0 and 1, higher values indicate better performance.
    '''
    if 4 * prec + sens == 0:
        return 0
    return (5 * prec * sens) / (4 * prec + sens)


def class_weights(binary: bool=False) -> torch.FloatTensor:
    '''Calculates the class weights according to their occurence in the dataset.

    Args:
        binary: Binary or multilabel segmentation.
    Returns:
        Tensor containing the class weights.
    '''
    df = calculate_statistics(binary=binary)
    arr = np.array(df.loc[:, 'Class Weight'], dtype=np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.FloatTensor(arr).to(device=device)
