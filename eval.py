import os
import torch
from models.unet import UNet
from PIL import Image
from torchvision import transforms


def eval_img(version: int, img_number: str):
    checkpoint_dir = '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/lightning_logs/UNet_model/version_' + str(version) + '/checkpoints'
    checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
    print(checkpoint)

    model = UNet.load_from_checkpoint(checkpoint)
    model.eval()

    img = Image.open('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung/lung_' + img_number + '.png')

    toTensor = transforms.ToTensor()
    x = toTensor(img)
    x = x.unsqueeze(0)

    y_hat = model(x)
    y_hat = y_hat.squeeze(0)
    y_pred = y_hat.argmax(dim=0).type(torch.FloatTensor)

    toPil = transforms.ToPILImage()
    img = toPil(y_pred)
    img.save('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/prediction/lung_' + img_number + '.png')


eval_img(version=34, img_number='018')