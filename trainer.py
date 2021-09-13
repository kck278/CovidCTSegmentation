import torch
import pytorch_lightning as pl
from typing import Optional
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.unet import UNet
from models.segnet import SegNet


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, images_dir: str, masks_dir: str, batch_size: int=8, num_classes: int=2):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dims = (1, 256, 256)


    def setup(self, stage: Optional[str]=None):
        # Assign datasets for use in dataloaders
        if stage == 'fit':
            covid_full = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir, num_classes=self.num_classes)
            train_len = int(0.72 * len(covid_full))
            val_len = int(0.10 * len(covid_full))
            test_len = len(covid_full) - (train_len + val_len)
            self.covid_train, self.covid_val, self.covid_test = random_split(covid_full, [train_len, val_len, test_len])


    def train_dataloader(self):
        return DataLoader(self.covid_train, batch_size=self.batch_size)


    def val_dataloader(self):
        return DataLoader(self.covid_val, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.covid_test, batch_size=1)


gpus = 1 if torch.cuda.is_available() else 0
logger = TensorBoardLogger('lightning_logs', name = 'UNet_model')
# logger = TensorBoardLogger('lightning_logs', name = 'SegNet_model')

# hyperparameters
num_classes = 4
epochs = 5
learning_rate = 5e-4
batch_size = 2

images_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung'

if num_classes == 2:
    masks_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/binary'
else:
    masks_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/multilabel'

covid_data_module = CovidDataModule(images_dir=images_dir, masks_dir=masks_dir, batch_size=batch_size, num_classes=num_classes)
model = UNet(num_classes=num_classes, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
# model = SegNet()

trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, log_every_n_steps=1, logger=logger)
trainer.fit(model, covid_data_module)
trainer.test(model)
