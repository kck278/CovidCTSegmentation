import os
import torch
from tqdm import tqdm
from models.unet import UNet
from models.unet_monai import UNetMonai
from models.segnet import SegNet
from models.segnet_original import SegNetOriginal
from PIL import Image
from torchvision import transforms
from typing import List, Optional
from data.util import sorted_alphanumeric


def eval_img(model_name: str, num_classes: int, version_name: str, image_name: Optional[str], resolution: int=256):
    versions_dir = os.path.join('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/lightning_logs', model_name)

    if version_name == 'latest':
        version_name = sorted_alphanumeric(os.listdir(versions_dir))[-1]

    checkpoint_dir = os.path.join(versions_dir, version_name, 'checkpoints')
    checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])

    if model_name == "UNet":
        model = UNet.load_from_checkpoint(checkpoint)
    elif model_name == "UNetMonai":
        model = UNetMonai.load_from_checkpoint(checkpoint)
    elif model_name == "SegNet":
        model = SegNet.load_from_checkpoint(checkpoint)
    elif model_name == "SegNetOriginal":
        model = SegNetOriginal.load_from_checkpoint(checkpoint)

    model.eval()

    img_dir = os.path.join('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung', str(resolution))

    if image_name is None:
        img_names = os.listdir(img_dir)
    else:
        img_names = [image_name]

    save_dir = os.path.join(
        '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/prediction/', 
        model_name, 
        version_name,
        'binary' if num_classes == 2 else 'multilabel',
        str(resolution)
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_name in tqdm(img_names):
        if img_name == "extended":
            continue
        
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('L')

        toTensor = transforms.ToTensor()
        x = toTensor(img)
        x = x.unsqueeze(0)

        y_hat = model(x)
        y_hat = y_hat.squeeze(0)
        y_pred = y_hat.argmax(dim=0).type(torch.FloatTensor)
        y_pred /= num_classes - 1

        save_path = os.path.join(save_dir, img_name)

        toPil = transforms.ToPILImage()
        img = toPil(y_pred)
        img.save(save_path)


eval_img(model_name='UNet', num_classes=4, version_name='version_1', image_name=None, resolution=256)
