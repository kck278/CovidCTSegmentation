import torch
import pytorch_lightning as pl
from typing import Optional
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.unet import UNet
from models.segnet import SegNet


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, 
                images_dir: str='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung',
                masks_dir: str='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/binary',
                batch_size: int=2,
                binary: bool=False):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size

        if not binary and masks_dir == '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/binary':
            self.masks_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/multilabel'

        self.dims = (1, 512, 512)


    def setup(self, stage: Optional[str]=None):
        # Assign datasets for use in dataloaders
        if stage == 'fit':
            covid_full = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir)
            train_len = int(0.72 * len(covid_full))
            val_len = int(0.10 * len(covid_full))
            test_len = len(covid_full) - (train_len + val_len)
            self.covid_train, self.covid_val, self.covid_test = random_split(covid_full, [train_len, val_len, test_len])


    def train_dataloader(self):
        return DataLoader(self.covid_train, batch_size=self.batch_size)


    def val_dataloader(self):
        return DataLoader(self.covid_val, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.covid_test, batch_size=self.batch_size)


gpus = 1 if torch.cuda.is_available() else 0
logger = TensorBoardLogger('lightning_logs', name = 'UNet_model')
# logger = TensorBoardLogger('lightning_logs', name = 'SegNet_model')

binary = True
covid_data_module = CovidDataModule(binary=binary)
model = UNet(binary=binary)
# model = SegNet()

trainer = pl.Trainer(max_epochs=160, gpus=gpus, progress_bar_refresh_rate=20, log_every_n_steps=1, logger=logger)
trainer.fit(model, covid_data_module)
trainer.test(model)
