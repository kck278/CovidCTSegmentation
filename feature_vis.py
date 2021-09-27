import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.unet import UNet
from models.unet_monai import UNetMonai
from models.segnet import SegNet
from models.segnet_original import SegNetOriginal
from PIL import Image
from torchvision import transforms
from typing import List, Optional
from data.util import sorted_alphanumeric


def define_model(model_name: str, version_name: str):
    versions_dir = os.path.join('lightning_logs', model_name)

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
    print(model)
    return model


def save_conv_parameters(model, model_name):
    model_weights = [] # save the conv layer weights in list
    conv_layers = [] # save the conv layers in list
    # get all model children as list
    model_children = list(model.children())
    # keep count of the conv layers
    counter = 0     

    # append all the conv layers and respective weights to list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    print(f"Total convolutional layers: {counter}")
    return model_weights, conv_layers


def show_conv_layers_and_weights(model_weights, conv_layers):
    for weight, conv in zip(model_weights, conv_layers):
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")


def create_path_save_images(model_name: str, version_name: str, num_classes: int, resolution: int):
    save_dir = os.path.join(
        'data/images/filter/', 
        model_name, 
        version_name,
        'binary' if num_classes == 2 else 'multi_class',
        str(resolution)
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filter_img_name = "filter_" + model_name
    path = os.path.join(save_dir, filter_img_name)
    return path, save_dir


def save_filter_image(model_weights, save_filter):
    plt.figure(figsize=(20, 17))

    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig(save_filter)

    plt.show()


def feature_vis(model_name: str, num_classes: int, version_name: str, image_name: Optional[str], resolution: int=256):
    model = define_model(model_name, version_name)
    model_weights, conv_layers = save_conv_parameters(model, model_name)
    show_conv_layers_and_weights(model_weights, conv_layers)
    save_filter, save_dir = create_path_save_images(model_name, version_name, num_classes, resolution)
    save_filter_image(model_weights, save_filter)
    
    # load image
    img_dir = os.path.join('data/images/png/lung', str(resolution))

    if image_name is None:
        img_names = os.listdir(img_dir)
    else:
        img_names = [image_name]


    for img_name in tqdm(img_names):
        if img_name == "extended":
            continue
        
        
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('L')
        
        # define the transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        img = np.array(img)
        # apply the transforms
        img = transform(img)
        print(img.size())
        # unsqueeze to add a batch dimension
        img = img.unsqueeze(0)
        print(img.size())
        print(conv_layers[0])

        # pass the image through all the layers
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results


        # visualize 64 features from each layer (might be more in upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())

            for i, filter in enumerate(layer_viz):
                if i == 64:
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter, cmap='gray')
                plt.axis("off")
                
            print(f"Saving layer {num_layer} feature maps...")
            save_path = os.path.join(save_dir, f"layer_{num_layer}.png")
            plt.savefig(save_path)
            plt.close()


feature_vis(model_name='SegNet', num_classes=2, version_name='version_95', image_name='lung_001.png', resolution=256)