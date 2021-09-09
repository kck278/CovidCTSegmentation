import torch
import pytorch_lightning as pl
from torch import nn
from monai.networks.nets import UNet
from metrics import calculate_metrics, class_weights, cross_entropy_loss


class UNet(pl.LightningModule):
    def __init__(self, binary: bool=False):
        super(UNet, self).__init__()
        self.weights = class_weights(binary=binary)

        # Encoder
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128, do_dropout=True)

        # Bottleneck
        self.bottleneck = ConvolutionBlock(128, 256)
        self.dropout = nn.Dropout2d(p=0.5)

        # Decoder
        self.decoder1 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)

        # Classifier
        self.conv = nn.Conv2d(64, 2, kernel_size=1, padding=0)
        self.softmax = nn.Softmax2d()


    def forward(self, x):
        # Encoder
        x1, p1 = self.encoder1(x)
        x2, p2 = self.encoder2(p1)

        # Bottleneck
        b1 = self.bottleneck(p2)
        b2 = self.dropout(b1)

        # Decoder
        d1 = self.decoder1(b2, x2)
        d2 = self.decoder2(d1, x1)

        # Classifier
        c = self.conv(d2)
        out = self.softmax(c)
        return out


    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy_loss(y_hat, y, self.weights)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.weights, 'val')
        self.log_dict(metrics)
        return metrics


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = calculate_metrics(y_hat, y, self.weights, 'test')
        self.log_dict(metrics)
        return metrics


    def configure_optimizers(self):
        # Adam: extension to stochastic gradient descent with optimizations
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, do_dropout=False):
        super().__init__()
        self.do_dropout = do_dropout
        self.conv = ConvolutionBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(p=0.5)
    

    def forward(self, inputs):
        x = self.conv(inputs)
        p = x

        if self.do_dropout:
            p = self.dropout(p)

        p = self.pool(p)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvolutionBlock(out_channels + out_channels, out_channels)
        self.relu = nn.ReLU()


    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = self.relu(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
