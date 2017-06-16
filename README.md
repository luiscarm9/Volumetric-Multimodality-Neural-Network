# Volumetric Multimodality Neural Network

Adaptation V-Net architecture for brain lesion segmentation in medical images with 4 different MRI modalities.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Programs and packages required to run our network:

```
SimpleITK 
Python2
Caffe-3D
CUDA with cuDNN
```
[SimpleITK installation](https://itk.org/Wiki/SimpleITK/GettingStarted#Build_It_Yourself) 
[Caffe-3D repo](https://github.com/faustomilletari/3D-Caffe)
 
### Training Process

The dataset mut be in a medical image format (.nii, .mhd, etc..)

The data must be in the following distribution:


```
Volumetric Multimodality Neural Network
│   README.md
│   main.py  
│   DataManager.py  
│   utilities.py  
│   layers.py
│   Vnet2.py    
└───Prototxt
│   │   test_noPooling_ResNet_cinque2.prototxt
│   │   train_noPooling_ResNet_cinque2.prototxt
└───Train
|   │   1_channel1.nii
|   │   1_channel2.nii
|   │   1_channel3.nii
|   │   1_channel4.nii
|   │   1_segmentation.nii
└───Test
|   │   1_channel1.nii
|   │   1_channel2.nii
|   │   1_channel3.nii
|   │   1_channel4.nii
```

Where channel# are the differents modlaities (Example: Falir, DWI, T1, T1c,...)
To run the network first set the training parameters in **main.py**. After seting paths to the files and training values run:
```
python main.py -train
```
The trained models will be saved on */Models/MRI_cinque_snapshots*
**Note:** This process uses a lot of memory we recomend using a GPU. For that you must have installed and set up cuDNN.
## Test Process

In order to train your model update the parameters in **main.py** to slect the last model or the iteration that want to be tested. After that only run:
```
python main.py -test
```
The output will be saved on the */Results* folder.
## Deployment

For additional information of how basic VNet architecture works look at this paper [VNet](https://arxiv.org/pdf/1606.04797v1.pdf) and the following tutorial [VNet tutorial](https://sagarhukkire.github.io/Vnet-Cafffe_Guide/)


## License

This project is licensed under the MIT License 

## Acknowledgments

* We thank V-Net and 3d-Caffe implementation by @faustomilletari. 

## Authors
* Silvana Castillo @SilvanaC
* Laura Daza @lauradaza
* Luis Rivera @luiscarm9
