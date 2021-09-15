import torch
import pytorch_lightning as pl
from typing import Optional
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.unet import UNet
from models.segnet import SegNet
from arguments import parse_arguments


class CovidDataModule(pl.LightningDataModule):
    def __init__(self, images_dir: str, masks_dir: str, num_classes: int=2, batch_size: int=8):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
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


args = parse_arguments()
model_name = args.model_name

gpus = 1 if torch.cuda.is_available() else 0
logger = TensorBoardLogger('lightning_logs', name=model_name)

# Hyperparameters
num_classes = args.num_classes
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size

images_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung'

if num_classes == 2:
    masks_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/binary'
else:
    masks_dir='/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/multilabel'

covid_data_module = CovidDataModule(
    images_dir=images_dir, 
    masks_dir=masks_dir, 
    num_classes=num_classes,
    batch_size=batch_size
)

if model_name == "UNet":
    model = UNet(num_classes=num_classes, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
else:
    model = SegNet(num_classes=num_classes, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, log_every_n_steps=1, logger=logger)
trainer.fit(model, covid_data_module)
trainer.test(model)
