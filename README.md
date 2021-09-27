# CovidCTSegmentation
This is the final project from the course "Advanced Machine Learning" at th University of Heidelberg in the summer semester 2021.

In this project, we use deep learning to segment Covid CT scans. The project aims to reproduce the paper ["COVID-19 lung CT image segmentation using deep learning methods: U-Net versus SegNet"](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-020-00529-5) by Adnan Saood and Iyad Hatem. Besides that, additional experiments were performed to further investigate the potential of deep learning in Covid Segmentation.

This readme should mainly explain how to run the code in our repository. For the theoretical background, a report has been written that explains the methods and results more in detail. If you are interested in the theoretical background and our results, please contact one of the team members.

In theory, the code runs on GPU as well as on CPU. However, we would strongly recommend to run the training on a GPU as it takes very long otherwise.

## Team Members
Kim-Celine Kahl  
ei260@stud.uni-heidelberg.de  

Miguel Heidegger  
tf268@stud.uni-heidelberg.de

Sophia Matthis  
cq270@stud.uni-heidelberg.de

## Setup
1. Clone the repository in your preferred directory
    ```bash
    git clone https://github.com/kck278/CovidCTSegmentation.git
    ```
<!--Install required packages Vielleicht sollten wir noch eine requirements.txt machen-->
2. (Optional) Download the dataset. We used the  [COVID-19 CT segmentation dataset](http://medicalsegmentation.com/covid19/). In the paper we tried to reproduce, only the first dataset on this website was used, which can be found in the section "Download data". However, we also implemented the possibility to extend the dataset with "Segmentation dataset nr. 2". The data and the preprocessed data is also pushed in this repository, so if you cloned the whole repository you might not need to execute this step.

## Preprocess the dataset

## Train a model

### Run training with a standard Train/ Validation/ Test split
Training a model is done by running file trainer.py. In the command line parameters can be passed in to determine the training parameter. 

    -m : neural network that is going to be trained. Choices: "UNet", "SegNet", "UNetMonai" or "SegNetOriginal", default: "UNet"

    -c : number of classes for segmentation. Binary segmentation mode (2) or multi-class segmentation mode (4), int, default 2

    -b : batch size, int, default 2

    -e : epochs for training, int, default 160

    -l : learning rate, float, default 1e-4

    -r : resolution of input images, either 256 or 512, int, default 256

    -ext : use extended dataset, bool, default False

exemplary command line for training:

```bash
python trainer.py -m "UNet" -c 2 -b 2 -l 1e-4
```

### Run 5-fold cross validation
Running 5-fold cross validation works analogous to running a normal training, but using the file trainer_5_fold_cross_validation.py

```bash
python trainer_5_fold_cross_validation.py -m "UNet" -c 2 -b 2 -l 1e-4
```

## Predict Segmentations
To actually predict segmentations in a visible way, the file eval.py needs to be run.
The desired network, number of classes, the version of the model, a specific image (if none is selected, all images are predicted) and resolution can be chosen by specifying them in the parameters of the method call.

```python
eval_img(model_name='UNet', num_classes=4, version_name='version_59', image_name=None, resolution=256)
```
## nnU-Net
The data to run the nnU-Net experiments can be found in the nnUNet folder. There, the raw data to run the experiments is in nnUNet_raw_data. Additionally, some trained models can be found in nnUNet_trained_models. 
To undestand the structure of the nnUNet folder, have a look at the [GitHub repository of the nnU-Net](https://github.com/MIC-DKFZ/nnUNet) where this structure is explained in detail.

The main experiments we performed are from Task503_CovidMulticlass and Task504_CovidBinary. 

If you want to run them with 1000 epochs (default nnU-Net configuration) you can just pip install nnU-Net as described on [GitHub](https://github.com/MIC-DKFZ/nnUNet). If you also want to train the models with 160 epochs or evaluate a model that was trained 160 epochs, you need to clone the repository according to the [instructions of nnU-Net, step 2.ii.](https://github.com/MIC-DKFZ/nnUNet#installation) Next, you need to create a file "nnUNetTrainerOwn.py" with the following code:

```
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

class nnUNetTrainerOwn(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 160
```
This file then needs to be stored in the nnUNet repository in nnunet/training/network_training.

Then you can run a model according to the [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet#2d-u-net). An example to train a model is

```bash
nnUNet_train 2d nnUNetTrainerV2 Task504_CovidBinary -f all
```

To evaluate the model, run 

```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t Task504_CovidBinary -m 2d -f all -tr nnUNetTrainerV2
```

If you want to run the multi-class experiment, replace Task504_CovidBinary by Task503_CovidMulticlass in these commands. If you want to use the nnUNetTrainerOwn for shorter training, replace nnUNetTrainerV2 by nnUNetTrainerOwn. 

## Visualize Feature Maps and Filter for SegNet
For visualizing the feature maps of SegNet, the file feature_vis.py should be run. If changes to the parameters are desired, they can be changed directly in the parameters of the method call. Right now the best model according to accuracy is selected in binary segmentation mode.

## Show the heatmaps of infected tissue
If you are interested in the distribution of the infected tissue, you can have a look at the png files in data/heatmap. If you want to reproduce these heatmaps, simply run heatmap.py. 