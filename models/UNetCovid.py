import torch
from torch import nn
from monai.networks.nets import UNet
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger

class UNetCovid(pl.LightningModule):

    def __init__(self):
        super(UNetCovid, self).__init__()
        self.model = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = self.cross_entropy_loss(y_hat,y)
        correct = y_hat.argmax(dim=1).eq(y).sum().item()
        tensorboard_logs = {'train_loss': loss}
        self.logger.experiment.add_scalar("Loss/Train", loss, batch_nb)
        self.logger.experiment.add_scalar("Correct/Train", correct, batch_nb)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': self.cross_entropy_loss(y_hat, y)}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss()
        # attention: this might crash when not using GPUs for training. In this case, remove .to(device='cuda')
        loss = criterion(logits, labels.type(torch.LongTensor).to(device='cuda'))
        return loss
