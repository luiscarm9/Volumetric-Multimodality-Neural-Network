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
        whichDataList = np.random.randint(1,15, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichDataForMatchingList = np.random.randint(1,15, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))

        for p,whichDataForMatching in zip(whichDataList,whichDataForMatchingList):

            currGtKey = str(p) + '_segmentation' + '.nii'
            currImgKey1 = str(p) + '_Flair' + '.nii'
            currImgKey2 = str(p) + '_T1' + '.nii'
            currImgKey3 = str(p) + '_DWI' + '.nii'
            currImgKey4 = str(p) + '_T2' + '.nii'

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

                #ImgGeneral = np.concatenate((defImg1,defImg2,defImg3,defImg4), axis = 2)
                #print "Prueba s1 " + str(ImgGeneral.shape)
                #ImgGeneral, defLab = utilities.produceRandomlyDeformedImage(ImgGeneral, defLab,self.params['ModelParams']['numcontrolpoints'],self.params['ModelParams']['sigma'])
                defImg1,defImg2,defImg3,defImg4,defLab = utilities.produceRandomlyDeformedImage(defImg1,defImg2,defImg3,defImg4,defLab,self.params['ModelParams']['numcontrolpoints'],self.params['ModelParams']['sigma'])

                #[defImg1,defImg2,defImg3,defImg4] = np.split(ImgGeneral,4,axis=2)
                #[defLab,ignorar1,ignorar2,ignorar3] = np.split(defLab,4,axis=2)


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

                #plt.clf()
                #plt.plot(range(0, it), train_loss[0:it])
                #plt.pause(0.00000001)
                with open('loss.txt', 'a') as t:
                    t.write(str(train_loss[it])+ "\n")

                t.close()
            #matplotlib.pyplot.show()


    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)

        #assert 4*howManyGT == howManyImages #AHORA TENEMOS 110 LABELS 440 DATA

        print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        test_interval = 10000
        # Write a temporary solver text file because pycaffe is stupid
        with open("solver.prototxt", 'w') as f:
            f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
            f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
            f.write("momentum: 0.90 \n")
            f.write("weight_decay: 0.0005 \n")
            f.write("lr_policy: \"fixed\" \n")
            #f.write("stepsize: 2000 \n")
            #f.write("gamma: 0.1 \n")
            f.write("display: 1 \n")
            f.write("snapshot: 1000 \n")
            f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
            #f.write("test_iter: 3 \n")
            #f.write("test_interval: " + str(test_interval) + "\n")

        f.close()
        solver = caffe.SGDSolver("solver.prototxt")
        os.remove("solver.prototxt")

        if (self.params['ModelParams']['snapshot'] > 0):
            solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(
                self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        numpyImages = self.dataManagerTrain.getNumpyImages()
        numpyGT = self.dataManagerTrain.getNumpyGT()

        #numpyImages['Case00.mhd']
        #numpy images is a dictionary that you index in this way (with filenames)

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key]-=mean
            numpyImages[key]/=std

        dataQueue = Queue(14) #max 50 images in queue
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
            #import pdb; pdb.set_trace()

        #for key in numpyGT:

            #sio.savemat('/mnt/tmp/ssd1/VNet-isles/VNet_ISLES_MODI/Results/'+(str(key))+'GT'+'.mat', {'vect':numpyGT[key]})

        results = dict()
        #pdb.set_trace()
        #for key in numpyImages:
        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()
        whichDataList = np.random.randint(1,15, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichDataForMatchingList = np.random.randint(1,15, size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        a1=np.arange(1,15)

        for key in a1:
            filename=str(key)
            Key1 = str(key) + '_Flair' + '.nii'
            Key2 = str(key) + '_T1' + '.nii'
            Key3 = str(key) + '_DWI' + '.nii'
            Key4 = str(key) + '_T2' + '.nii'
            btch = np.zeros((1, 4,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]),dtype=float)


            btch[:, 0, :, :, :] = np.reshape(numpyImages[Key1],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 1, :, :, :] = np.reshape(numpyImages[Key2],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 2, :, :, :] = np.reshape(numpyImages[Key3],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])
            btch[:, 3, :, :, :] = np.reshape(numpyImages[Key4],[1,1,numpyImages[Key1].shape[0],numpyImages[Key1].shape[1],numpyImages[Key1].shape[2]])

            net.blobs['data'].data[...] = btch

            out = net.forward()

            #l1 = out["labelmap"]

            l=out["labelmapsf"]

            arr = np.zeros(shape=(1,1,2973696))
            arr[0,0,:]=l[:,1,:]

            #labelmap1 = np.squeeze(l1[0,1,:,:,:])
            labelmap=l

            results[key] = np.squeeze(labelmap)
            #results[key] = np.squeeze(labelmap1)
            sio.savemat('/mnt/tmp/ssd1/VNet-isles/VNet_ISLES_MODI/Results/'+(filename)+'.mat', {'vect':arr})

            #self.dataManagerTest.writeResultsFromNumpyLabel(np.squeeze(labelmap1),Key1)
