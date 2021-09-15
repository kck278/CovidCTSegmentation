import os
import torch
from models.unet import UNet
from models.segnet import SegNet
from PIL import Image
from torchvision import transforms


def eval_img(model_name: str, num_classes: int, version_name: str, img_number: str):
    versions_dir = '/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/lightning_logs/' + model_name + '/'

    if version_name == 'latest':
        version_name = sorted(os.listdir(versions_dir))[-1]

    checkpoint_dir = versions_dir + version_name + '/checkpoints'
    checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])

    if model_name == 'UNet':
        model = UNet.load_from_checkpoint(checkpoint)
    else:
        model = SegNet.load_from_checkpoint(checkpoint)

    model.eval()

    img = Image.open('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/png/lung/lung_' + img_number + '.png')

    toTensor = transforms.ToTensor()
    x = toTensor(img)
    x = x.unsqueeze(0)

    y_hat = model(x)
    y_hat = y_hat.squeeze(0)
    y_pred = y_hat.argmax(dim=0).type(torch.FloatTensor)
    y_pred /= num_classes - 1

    toPil = transforms.ToPILImage()
    img = toPil(y_pred)
    img.save('/home/hd/hd_hd/hd_ei260/CovidCTSegmentation/data/images/prediction/' + model_name + '/lung_' + img_number + '.png')


eval_img(model_name='UNet', num_classes=4, version_name='latest', img_number='011')
#eval_img(model_name='SegNet', num_classes=2, version_name='full_run/version_9', img_number='017')
