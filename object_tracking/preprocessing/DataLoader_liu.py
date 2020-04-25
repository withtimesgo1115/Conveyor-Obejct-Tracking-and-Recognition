import cv2
import numpy as np
import math
import os
import random
import glob

class DataLoader_liu():
    def __init__(self,path,idx=0,cvt=None,size=None):
        # here use super to make this class succeed itself
        super(DataLoader_liu).__init__()
        # constructor
        self.path = path
        self.idx = idx
        self.cvt = cvt
        self.size = size
        self._len = len(glob.glob(path+'*.png'))

    # define this member function to handle basic img transformation
    def cvtImg(self,img):
        if isinstance(self.cvt,np.int):
            img = cv2.cvtColor(img,self.cvt)
        if isinstance(self.size,tuple):
            img = cv2.resize(img,self.size)
        return img
    
    # define this member function to get Img by using index
    def getItem(self,idx):
        img = self.cvtImg(cv2.imread(self.path+str(idx)+'.png'))
        return img
    
    # define this member function to call getItem and increase idx by 1
    # but in my view, this function is not necessary, we can put self.idx+=1 in the last funtion 
    def __next__(self):
        img = self.getItem(self.idx)
        self.idx += 1
        return img
    
    # define this member function to acquire a img list from index to the end of path folder
    # this function is based on former functions too
    def getRest(self):
        img = []
        for i in range(self.idx,self._len):
            img.append(self.__next__())
        return img

    # define a property here to acquire the length of folder storing images
    # the goal of this property is to get the info in a convenient way
    # because we can just call object.len() instaed of writing a long sentence
    @property
    def len(self):
        return self._len




