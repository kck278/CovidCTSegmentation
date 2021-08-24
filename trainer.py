import pytorch_lightning as pl
from dataset import CovidDataset
from torch.utils.data import random_split, DataLoader
from models import UNetCovid

class CovidDataModule(pl.LightningDataModule):
    def __init__(self, images_dir="/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/lung",
                 masks_dir="/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/binary_mask",
                 batch_size=12):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size

        self.dims = (1, 512, 512)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir)
            self.covid_train, self.covid_val = random_split(mnist_full, [80, 20])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.covid_test = CovidDataset(images_dir=self.images_dir, masks_dir=self.masks_dir)

    def train_dataloader(self):
        return DataLoader(self.covid_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.covid_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.covid_test, batch_size=self.batch_size)


dm = CovidDataModule()
model = UNetCovid()
trainer = pl.Trainer(max_epochs=1, gpus=0, progress_bar_refresh_rate=20)
trainer.fit(model, dm)
