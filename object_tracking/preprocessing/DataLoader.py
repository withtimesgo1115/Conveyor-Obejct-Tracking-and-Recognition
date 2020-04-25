import cv2
import numpy as np
import math
import os
import glob
import random

class DataLoader():
    def __init__(self, path, idx=0, cvt=None, size=None):
        super(DataLoader).__init__()
        self.path = path
        self.cvt = cvt
        self.idx = idx
        self.size = size
        self._len = len(glob.glob(path+'*.png'))

    def cvtImg(self, im):
        if isinstance(self.cvt, np.int):
            im = cv2.cvtColor(im, self.cvt)
        if isinstance(self.size, tuple):
            im = cv2.resize(im, self.size)
        return im

    def getItem(self, idx):
        im = self.cvtImg(cv2.imread(self.path+str(idx)+'.png'))
        return im
    
    def __next__(self):
        im = self.getItem(self.idx)
        self.idx+=1
        return im
    
    def getRest(self):
        img = []
        for i in range(self.idx, self.len):
            img.append(self.__next__())
        return img
        #return [self.__next__() for i in range(self.idx, self.len)]

    @property
    def len(self):
        return self._len