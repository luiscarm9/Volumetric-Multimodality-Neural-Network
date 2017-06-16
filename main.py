import sys
import os
import numpy as np
import VNet2 as VN
import pdb


basePath=os.getcwd()

params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
params['ModelParams']['numcontrolpoints']=2
params['ModelParams']['sigma']=15
params['ModelParams']['device']=0 #GPU id to use
params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/train_noPooling_ResNet_cinque2.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/test_noPooling_ResNet_cinque2.prototxt')
params['ModelParams']['snapshot']=10 #model iteration numer to resume training or test
params['ModelParams']['dirTrain']=os.path.join(basePath,'/mnt/tmp/home4/brats/Vnet/Dataset/SISS/TRAIN')
params['ModelParams']['dirTest']=os.path.join(basePath,'/mnt/tmp/home4/brats/Vnet/Dataset/SISS/TEST')
params['ModelParams']['dirResult']=os.path.join(basePath,'Results') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'Models/MRI_cinque_snapshots/') #where to save the models while training
params['ModelParams']['batchsize'] = 1 #the batchsize
params['ModelParams']['numIterations'] = 500000 #the number of training iterations
params['ModelParams']['baseLR'] = 0.00001 #the learning rate, initial one [Check VNet2.py if want to change the lr policy]
params['ModelParams']['nProc'] = 1 #the number of threads to do data augmentation

#Modalities names and image format
params['ModelParams']['mod1']='Flair'
params['ModelParams']['mod2']='DWI'
params['ModelParams']['mod3']='T1'
params['ModelParams']['mod4']='T2'
params['ModelParams']['format']='.nii'

#Size of Train/Test dataset
params['ModelParams']['TrainSize']=14
params['ModelParams']['TestSize']=14



#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1,1,1],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([176,176,96],dtype=int)
params['DataManagerParams']['normDir'] = False
params['DataManagerParams']['TrainSize']=14 
params['DataManagerParams']['format']='.nii'

model=VN.VNet(params)
train = [i for i, j in enumerate(sys.argv) if j == '-train']
if len(train)>0:
    model.train()

test = [i for i, j in enumerate(sys.argv) if j == '-test']
if len(test) > 0:
    model.test()
