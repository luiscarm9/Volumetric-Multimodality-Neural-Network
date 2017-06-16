import sys
sys.path
sys.path.append('/mnt/tmp/ssd1/3D-Caffe/python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import DataManager as DM
import utilities
from os.path import splitext
from multiprocessing import Process, Queue
import scipy.io as sio
import pdb

class VNet(object):
    params=None
    dataManagerTrain=None
    dataManagerTest=None

    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()

    def prepareDataThread(self, dataQueue, numpyImages, numpyGT):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        keysIMG = numpyImages.keys()

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()
        whichDataList = np.random.randint(1,int(self.params['ModelParams']['TrainSize'])+1, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichDataForMatchingList = np.random.randint(1,int(self.params['ModelParams']['TrainSize'])+1, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))

        for p,whichDataForMatching in zip(whichDataList,whichDataForMatchingList):

            currGtKey = str(p) + '_segmentation' + '.nii'
            currImgKey1 = str(p) +'_'+ self.params['ModelParams']['mod1'] + self.params['ModelParams']['format']
            currImgKey2 = str(p) +'_'+ self.params['ModelParams']['mod2'] + self.params['ModelParams']['format']
            currImgKey3 = str(p) +'_'+ self.params['ModelParams']['mod3'] + self.params['ModelParams']['format']
            currImgKey4 = str(p) +'_'+ self.params['ModelParams']['mod4'] + self.params['ModelParams']['format']

            defImg1= numpyImages[currImgKey1]
            defImg2= numpyImages[currImgKey2]
            defImg3= numpyImages[currImgKey3]
            defImg4= numpyImages[currImgKey4]
            defLab = numpyGT[currGtKey]

            ImgKeyMatching = keysIMG[whichDataForMatching]

            defImg1 = utilities.hist_match(defImg1, numpyImages[ImgKeyMatching])
            defImg2 = utilities.hist_match(defImg2, numpyImages[ImgKeyMatching])
            defImg3 = utilities.hist_match(defImg3, numpyImages[ImgKeyMatching])
            defImg4 = utilities.hist_match(defImg4, numpyImages[ImgKeyMatching])

            if(np.random.rand(1)[0]>0.5):
                defImg1,defImg2,defImg3,defImg4,defLab = utilities.produceRandomlyDeformedImage(defImg1,defImg2,defImg3,defImg4,defLab,self.params['ModelParams']['numcontrolpoints'],self.params['ModelParams']['sigma'])

            weightData = np.zeros_like(defLab,dtype=float)
            weightData[defLab == 1] = np.prod(defLab.shape) / np.sum((defLab==1).astype(dtype=np.float32))
            weightData[defLab == 0] = np.prod(defLab.shape) / np.sum((defLab == 0).astype(dtype=np.float32))

            dataQueue.put(tuple((defImg1,defImg2,defImg3,defImg4,defLab,weightData)))

    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        batchData = np.zeros((batchsize, 4, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        #only used if you do weighted multinomial logistic regression
        batchWeight = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0],
                               self.params['DataManagerParams']['VolSize'][1],
                               self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        train_loss = np.zeros(nr_iter)
        for it in range(nr_iter):
            for i in range(batchsize):
                [defImg1,defImg2,defImg3,defImg4,defLab,defWeight] = dataQueue.get()
                #pdb.set_trace()
                batchData[i, 0, :, :, :] = defImg1.astype(dtype=np.float32)
                batchData[i, 1, :, :, :] = defImg2.astype(dtype=np.float32)
                batchData[i, 2, :, :, :] = defImg3.astype(dtype=np.float32)
                batchData[i, 3, :, :, :] = defImg4.astype(dtype=np.float32)
                batchLabel[i, 0, :, :, :] = (defLab > 0.5).astype(dtype=np.float32)
                batchWeight[i, 0, :, :, :] = defWeight.astype(dtype=np.float32)

            solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            #solver.net.blobs['labelWeight'].data[...] = batchWeight.astype(dtype=np.float32)
            #use only if you do softmax with loss


            solver.step(1)  # this does the training
            train_loss[it] = solver.net.blobs['loss'].data

            if (np.mod(it, 1) == 0):
                with open('loss.txt', 'a') as t:
                    t.write(str(train_loss[it])+ "\n")
                t.close()


    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)


        print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        test_interval = 10000
        with open("solver.prototxt", 'w') as f:
            f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
            f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
            f.write("momentum: 0.90 \n")
            f.write("weight_decay: 0.0005 \n")
            f.write("lr_policy: \"fixed\" \n")
            f.write("display: 1 \n")
            f.write("snapshot: 10 \n")
            f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
        f.close()
        solver = caffe.SGDSolver("solver.prototxt")
        os.remove("solver.prototxt")

        if (self.params['ModelParams']['snapshot'] > 0):
            solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(
                self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        numpyImages = self.dataManagerTrain.getNumpyImages()
        numpyGT = self.dataManagerTrain.getNumpyGT()


        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key]-=mean
            numpyImages[key]/=std

        dataQueue = Queue(int(self.params['ModelParams']['TrainSize'])) #Change if memory is not enough for all dataset
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver)


    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        #self.dataManagerTest.loadTestData()
        self.dataManagerTest.loadTrainingData()

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()
        numpyGT = self.dataManagerTest.getNumpyGT()

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])
            numpyImages[key] -= mean
            numpyImages[key] /= std

        results = dict()
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()
        whichDataList = np.random.randint(1,int(self.params['ModelParams']['TestSize'])+1, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichDataForMatchingList = np.random.randint(1,int(self.params['ModelParams']['TestSize'])+1, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        a1=np.arange(1,int(self.params['ModelParams']['TestSize'])+1)

        for key in a1:
            filename=str(key)
            Key1 = str(key) + '_'+ self.params['ModelParams']['mod1'] + self.params['ModelParams']['format']
            Key2 = str(key) + '_'+ self.params['ModelParams']['mod2'] + self.params['ModelParams']['format']
            Key3 = str(key) + '_'+ self.params['ModelParams']['mod3'] + self.params['ModelParams']['format']
            Key4 = str(key) + '_'+ self.params['ModelParams']['mod4'] + self.params['ModelParams']['format']
            btch = np.zeros((1, 4,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]),dtype=float)


            btch[:, 0, :, :, :] = np.reshape(numpyImages[Key1],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 1, :, :, :] = np.reshape(numpyImages[Key2],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 2, :, :, :] = np.reshape(numpyImages[Key3],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 3, :, :, :] = np.reshape(numpyImages[Key4],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])

            net.blobs['data'].data[...] = btch

            out = net.forward()

            l1 = out["labelmap"]

            labelmap1 = np.squeeze(l1[0,1,:,:,:])
            results[key] = np.squeeze(labelmap1)

            self.dataManagerTest.writeResultsFromNumpyLabel((labelmap1),Key1)
