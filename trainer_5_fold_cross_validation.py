import os
import torch
import pytorch_lightning as pl

from torch.utils.data.dataset import Subset
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.unet import UNet
from models.segnet import SegNet
from models.unet_monai import UNetMonai
from models.segnet_original import SegNetOriginal
from arguments import parse_arguments
from sklearn.model_selection import KFold
import numpy as np


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, images_dir: str, masks_dir: str, num_classes: int=2, batch_size: int=8, extended: bool=False):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dims = (1, 256, 256)
        self.extended = extended


    def train_dataloader(self, train_data):
        return DataLoader(train_data, batch_size=self.batch_size)


    def val_dataloader(self):
        return DataLoader(self.covid_val, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.covid_test, batch_size=1)


args = parse_arguments()
model_name = args.model_name

gpus = 1 if torch.cuda.is_available() else 0
name = model_name + "_5_fold"
logger = TensorBoardLogger('lightning_logs', name=name)

# Hyperparameters
num_classes = args.num_classes
resolution = args.resolution
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
extended = args.extended

images_dir = os.path.join('data/images/png/lung', str(resolution))
mask_dir = 'data/images/png/mask'

if num_classes == 2:
    masks_dir = os.path.join(mask_dir, 'binary', str(resolution))
else:
    masks_dir = os.path.join(mask_dir, 'multi_class', str(resolution))

if model_name == "UNet":
    model = UNet(
        num_classes=num_classes, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        resolution=resolution,
        extended=extended
    )
elif model_name == "UNetMonai":
    model = UNetMonai(
        num_classes=num_classes, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        resolution=resolution,
        extended=extended
    )
elif model_name == "SegNet":
    model = SegNet(
        num_classes=num_classes, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        resolution=resolution,
        extended=extended
    )
elif model_name == "SegNetOriginal":
    model = SegNetOriginal(
        num_classes=num_classes, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        batch_size=batch_size,
        resolution=resolution,
        extended=extended
    )


covid_full = CovidDataset(images_dir=images_dir, masks_dir=masks_dir, num_classes=num_classes, extended=extended)
k = 5
kf = KFold(n_splits=k, shuffle=True)

test_results = []

for train_idx, test_idx in kf.split(covid_full):
    train_data = Subset(covid_full, train_idx)
    test_data = Subset(covid_full, test_idx)
    
    # 10% validation - 1/8 of train_data
    val_len = int(0.125 * len(train_data))
    train_len = len(train_data) - (val_len)
    val_data, train_data = random_split(train_data, [val_len, train_len])


    trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, log_every_n_steps=1, logger=logger)
    
    #Create Dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=1)
    validation_loader = DataLoader(train_data, batch_size=batch_size)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=validation_loader)
    results = trainer.test(model, test_dataloaders=test_loader)
    test_results.append(results[0])

metrics_dict = {}

for i in range(1, num_classes):
    # calculate sensitivity
    mean_sensitivity = np.mean(np.array([res['test_sens_c' + str(i)] for res in test_results]))
    std_dev_sensitivity = np.std(np.array([res['test_sens_c' + str(i)] for res in test_results]))

    # calculate specificity
    mean_specificity = np.mean(np.array([res['test_spec_c' + str(i)] for res in test_results]))
    std_dev_specificity = np.std(np.array([res['test_spec_c' + str(i)] for res in test_results]))

    # calculate dice
    mean_dice = np.mean(np.array([res['test_dice_c' + str(i)] for res in test_results]))
    std_dev_dice = np.std(np.array([res['test_dice_c' + str(i)] for res in test_results]))

    # calculate g-mean
    mean_gmean = np.mean(np.array([res['test_mean_c' + str(i)] for res in test_results]))
    std_dev_gmean = np.std(np.array([res['test_mean_c' + str(i)] for res in test_results]))

    # calculate f2
    mean_f2 = np.mean(np.array([res['test_f2_c' + str(i)] for res in test_results]))
    std_dev_f2 = np.std(np.array([res['test_f2_c' + str(i)] for res in test_results]))

    metrics_dict["5_fold_mean_sensitivity_c" + str(i)] = mean_sensitivity
    metrics_dict["5_fold_std_dev_sensitivity_c" + str(i)] = std_dev_sensitivity
    metrics_dict["5_fold_mean_specificity_c" + str(i)] = mean_specificity
    metrics_dict["5_fold_std_dev_specificity_c" + str(i)] = std_dev_specificity
    metrics_dict["5_fold_mean_dice_c" + str(i)] = mean_dice
    metrics_dict["5_fold_std_dev_dice_c" + str(i)] = std_dev_dice
    metrics_dict["5_fold_mean_gmean_c" + str(i)] = mean_gmean
    metrics_dict["5_fold_std_dev_gmean_c" + str(i)] = std_dev_gmean
    metrics_dict["5_fold_mean_f2_c" + str(i)] = mean_f2
    metrics_dict["5_fold_std_dev_f2_c" + str(i)] = std_dev_f2
print(metrics_dict)
logger.log_metrics(metrics_dict)
logger.save()