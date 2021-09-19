import torch
import numpy as np
import pytorch_lightning as pl
from monai.networks.nets import UNet as MonaiUNet
from metrics import calculate_accuracy_metrics, calculate_metrics, class_weights, cross_entropy_loss


class UNetMonai(pl.LightningModule):
    def __init__(self, num_classes: int, epochs: int, learning_rate: float, batch_size: int, resolution: int, extended: bool):
        super(UNetMonai, self).__init__()
        self.save_hyperparameters()

        self.model = MonaiUNet(
            dimensions=2,
            in_channels=1,
            out_channels=num_classes,
            channels=(1, 64, 128, 256),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            num_res_units=2,
            dropout=0.5
        )

        self.class_weights = class_weights(num_classes=num_classes, extended=extended)
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.resolution = resolution
        self.extended = extended


    def forward(self, x):
        out = self.model(x)
        return out


    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, 
            { 
                "hp/epochs": self.epochs, 
                "hp/learning_rate": self.learning_rate,
                "hp/batch_size": self.batch_size,
                "hp/num_classes": self.num_classes,
            }
        )


    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.num_classes, self.class_weights, 'train')
        self.log_dict(metrics, sync_dist=True)
        return metrics['train_loss']


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.num_classes, self.class_weights, 'val')
        self.log_dict(metrics, sync_dist=True)
        return metrics


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.num_classes, self.class_weights, 'test')
        self.log_dict(metrics, sync_dist=True)
        return metrics


    def test_epoch_end(self, outputs) -> None:
        accuracies = np.array([out['test_acc'].cpu() for out in outputs])
        metrics = calculate_accuracy_metrics(accuracies)
        self.log_dict(metrics, sync_dist=True)
        return metrics


    def configure_optimizers(self):
        # Adam: extension to stochastic gradient descent with optimizations
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            gamma=0.95,
            step_size=12
        )
        return [optimizer], [scheduler]
