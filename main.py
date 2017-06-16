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
params['ModelParams']['device']=0
params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/train_noPooling_ResNet_cinque2.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/test_noPooling_ResNet_cinque2.prototxt')
params['ModelParams']['snapshot']=0
params['ModelParams']['dirTrain']=os.path.join(basePath,'/mnt/tmp/home4/brats/Vnet/Dataset/SISS/TRAIN')
params['ModelParams']['dirTest']=os.path.join(basePath,'/mnt/tmp/home4/brats/Vnet/Dataset/SISS/TEST')
params['ModelParams']['dirResult']=os.path.join(basePath,'Results') #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(basePath,'Models/MRI_cinque_snapshots/') #where to save the models while training
params['ModelParams']['batchsize'] = 1 #the batchsize
params['ModelParams']['numIterations'] = 500000#the number of iterations
params['ModelParams']['baseLR'] = 0.00001 #the learning rate, initial one
params['ModelParams']['nProc'] = 1 #the number of threads to do data augmentation


#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1,1,1],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([176,176,96],dtype=int)
params['DataManagerParams']['normDir'] = False 

model=VN.VNet(params)
train = [i for i, j in enumerate(sys.argv) if j == '-train']
if len(train)>0:
    model.train()

test = [i for i, j in enumerate(sys.argv) if j == '-test']
if len(test) > 0:
    model.test()
