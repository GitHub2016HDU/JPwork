#!/usr/bin/evn python
# -*- coding: utf-8 -*-

from os import walk
from os.path import join
import numpy as np
import tensorflow as tf
import os
import random
import cv2


class dataprocess(object):
    def __int__(self, datapath, batchsize, num_per_epoch, is_shuffle):
        self.datapath       = datapath
        self.batchsize      = batchsize
        self.num_per_epoch  = num_per_epoch
        self.is_shuffle     = is_shuffle

    def _getOneSet(self, setPath, setName, other):
        classes = {}
        classes_index = 0
        nameAppendlabel = []
        for root, dirs, files in walk(setPath):
            if len(dirs) !=0 :
                for d in dirs:
                    if 'true' in d:
                        classes[d] = [0, 1]
                    else:
                        classes[d] = [1, 0]
            for f in files:
                path = join(root, f)
                for key in classes:
                    if key in path:
                        nameAppendlabel.append([path, classes[key]])
        print('{} has {} cubic!'.format(setName, len(nameAppendlabel)))
        return nameAppendlabel

    def runAllSet(self, dataPath, setSlected, isShuffle):
        allNameAppendLabel = []
        allNameAppendLabel01 = []
        for num in setSlected:
            setName = 'subset'+str(num)
            setPath = join(dataPath, setName)
            onesetNameAppendlabel = self._getOneSet(setPath=setPath, setName=setName, other=None)
            if isShuffle == True:
                random.shuffle(onesetNameAppendlabel)
                allNameAppendLabel01[len(allNameAppendLabel01):len(allNameAppendLabel01)] = onesetNameAppendlabel
                random.shuffle(allNameAppendLabel01)
        print('all of cubic in subset has {}'.format(len(allNameAppendLabel01)))
        return allNameAppendLabel01

    def fromNameAppendLableGetSetNameLabel(self, NameAppendlabels):
        allNames = []
        allLables = []
        for s_i in range(len(NameAppendlabels)):
            try:
                allNames.append(NameAppendlabels[s_i][0])
                allLables.append(NameAppendlabels[s_i][1])
            except Exception as err:
                print(err)
            finally:
                pass
        return allNames, allLables

    def LoadNpDate(self, BatchName):
        BatchData = []
        try:
            for b_nam in BatchName:
                BatchData.append(np.load(b_nam))
        except Exception as err:
            print(err)
        finally:
            pass
        return np.array(BatchData)

    def BatchConfusionMatrix(self, BatchName, BatchLabel, BatchData, BatchPred, Threshold, test_batch_size, Path2Save):
        BTP, BFN, BTN, BFP = 0, 0, 0, 0
        for b in range(test_batch_size):
            if BatchLabel[b][1] == 1:
                if BatchPred[b][1] > Threshold:
                    BTP += 1
                else:
                    BFN += 1
            if BatchLabel[b][1] == 0:
                if BatchPred[b][0] > Threshold:
                    BTN += 1
                else:
                    BFP += 1
        return BTP, BFN, BTN, BFP

    def BatchConfusionMatrixPrintPnf(self, BatchName, BatchLabel, BatchData, BatchPred, Threshold, test_batch_size, Path2Save):
        BTP, BFN, BTN, BFP = 0, 0, 0, 0
        for b in range(test_batch_size):
            if BatchLabel[b][1] == 1:
                if BatchPred[b][1] > Threshold:
                    BTP += 1
                else:
                    BFN += 1
                    nametr2fl = BatchName[b]
                    path4tr2flSave = './{}/{}'.format(Path2Save, 'real-' + str(float('%.20f' % BatchPred[b][1])) + str(nametr2fl[0]))
                if BatchPred[b][1] < 0.9:
                    nametr2fl = BatchName[b]
                    datatr2fl = np.load(nametr2fl)
                    path4tr2flSave = './{}/{}'.format(Path2Save, 'real-' + str(float('%.20f' % BatchPred[b][1])) + str(nametr2fl[0]))
                    if not os.path.exists(path4tr2flSave):
                        os.makedirs(path4tr2flSave)
                    for sl in range(datatr2fl.shape[0]):
                        cv2.imwrite(os.path.join(path4tr2flSave, str(sl) + '.png'), datatr2fl[sl]*255)
                        pass
            if BatchLabel[b][1] == 0:
                if BatchPred[b][0] > Threshold:
                    BTN += 1
                else:
                    BFP += 1
                if BatchPred[b][0] < 0.95:
                    nametr2fl = BatchName[b]
                    datatr2fl = np.load(nametr2fl)
                    path4tr2flSave = './{}/{}'.format(Path2Save,
                                                      'false-' + str(float('%.20f' % BatchPred[b][1])) + str(nametr2fl[0]))
                    if not os.path.exists(path4tr2flSave):
                        os.makedirs(path4tr2flSave)
                    for sl in range(datatr2fl.shape[0]):
                        cv2.imwrite(os.path.join(path4tr2flSave, str(sl) + '.png'), datatr2fl[sl] * 255)
                        pass
        return BTP, BFN, BTN, BFP

    def fromNameGetCoodinate(self, fileName):
        try:
            cood_part = fileName[fileName.index('p') + 3:fileName.index('z') - 2]
            cood = []
            for cod in cood_part.split('_'):
                cood.append(float(cod))
            name_part = fileName[fileName.index('s')+2:fileName.index('h')-2]
            return name_part, cood
        except Exception as err:
            print(err)
        finally:
            pass


























