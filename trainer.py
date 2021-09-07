import pytorch_lightning as pl
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from models.UNetCovid import UNetCovid
from models.SegNet import SegNet

class CovidDataModule(pl.LightningDataModule):
    def __init__(self, images_dir="/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung",
                 masks_dir="/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/mask/binary",
                 batch_size=12):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size

        self.dims = (1, 512, 512)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            covid_full = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir)
            self.covid_train, self.covid_val, self.covid_test = random_split(covid_full, [72, 10, 18])

        # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.covid_test = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir)

    def train_dataloader(self):
        return DataLoader(self.covid_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.covid_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.covid_test, batch_size=self.batch_size)


logger = TensorBoardLogger('lightning_logs', name = 'UNet_model')
# logger = TensorBoardLogger('lightning_logs', name = 'SegNet_model')
dm = CovidDataModule()
model = UNetCovid()
# model = SegNet()
trainer = pl.Trainer(max_epochs=5, gpus=1, progress_bar_refresh_rate=20, log_every_n_steps=1, logger=logger)
trainer.fit(model, dm)
