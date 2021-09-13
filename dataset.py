import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader


class CovidDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, num_classes: int):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.num_classes = num_classes
        self.transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Lambda(self.multiply)])
        self.image_names, self.mask_names = self.make_dataset()


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        mask_name = self.mask_names[index]
        image = default_loader(os.path.join(self.images_dir, image_name))
        mask = default_loader(os.path.join(self.masks_dir, mask_name))
        assert image.size == mask.size
        image = self.transforms(image)
        mask = self.transforms(mask)
        mask = torch.squeeze(mask)
        return image, mask


    def make_dataset(self):
        image_names = sorted(os.listdir(self.images_dir))
        mask_names = sorted(os.listdir(self.masks_dir))
        assert len(image_names) == len(mask_names)
        return image_names, mask_names

    def multiply(self, tensor):
        return torch.mul(tensor, self.num_classes - 1)