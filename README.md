# Volumetric Multimodality Neural Network

Adaptation V-Net architecture for brain lesion segmentation in medical images with 4 different MRI modalities.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Programs and packages required to run our network:

```
SimpleITK ([link](https://itk.org/Wiki/SimpleITK/GettingStarted#Build_It_Yourself) )
Python2
Caffe-3D ([Caffe3D repo](https://github.com/faustomilletari/3D-Caffe))
CUDA with CudNN
```

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
│   Vnet2.py    │
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



End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system


## License

This project is licensed under the MIT License 

## Acknowledgments

* We thank V-Net and 3d-Caffe implementation by @faustomilletari. 

## Authors
* Silvana Castillo @SilvanaX
* Laura Daza @lauradaza
* Luis Rivera @luiscarm9
