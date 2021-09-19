import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import torchmetrics
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from metrics import calculate_accuracy_metrics, calculate_metrics, class_weights, cross_entropy_loss


class SegNet(pl.LightningModule):
    def __init__(self, num_classes: int, epochs: int, learning_rate: float, batch_size: int, resolution: int, extended: bool):
        super(SegNet, self).__init__()
        self.save_hyperparameters()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
       
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(num_classes, momentum=batchNorm_momentum)
        
        self.softmax = nn.Softmax2d()

        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxunpool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.class_weights = class_weights(num_classes=num_classes, extended=extended)
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.resolution = resolution
        self.extended = extended

    def forward(self, x):
        # Stage 1e
        x11 = self.relu(self.bn11(self.conv11(x)))
        x12 = self.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = self.maxpool2d(x12)

        # Stage 2e
        x21 = self.relu(self.bn21(self.conv21(x1p)))
        x22 = self.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = self.maxpool2d(x22)

        # Stage 3e
        x31 = self.relu(self.bn31(self.conv31(x2p)))
        x32 = self.relu(self.bn32(self.conv32(x31)))
        x3p, id3 = self.maxpool2d(x32)

        # Stage 3d
        x3d = self.maxunpool2d(x3p, id3)
        x32d = self.relu(self.bn32d(self.conv32d(x3d)))
        x31d = self.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.maxunpool2d(x31d, id2)
        x22d = self.relu(self.bn22d(self.conv22d(x2d)))
        x21d = self.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x3d = self.maxunpool2d(x21d, id1)
        x33d = self.relu(self.bn12d(self.conv12d(x3d)))
        x32d = self.relu(self.bn11d(self.conv11d(x33d)))

        xsoftmax = self.softmax(x32d)

        return xsoftmax

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy_loss(y_hat, y, self.class_weights)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.num_classes, self.class_weights, 'val')
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.num_classes, self.class_weights, 'test')
        self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs) -> None:
        accuracies = np.array([out['test_acc'] for out in outputs])
        metrics = calculate_accuracy_metrics(accuracies)
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


