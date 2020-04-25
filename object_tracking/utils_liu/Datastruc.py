import numpy as np
from numpy import linalg
import random

# here we'd better define a queue class to handle data
# queue is a simple data structure and its size is a basic parameter that we should consider 
class Queue(object):
    def __init__(self,size):
        super(Queue,self).__init__()
        self.size = size
        # define a list to store the data
        # we can learn that all the data structure(ADTs) can be achieved by using built-in data sturcture and methods  
        self.data = []
    
    # define a function to add element to the queue
    # what we should notice is the size must be larger than current data length
    def push(self,d):
        if len(self.data)<self.size:
            self.data.append(d)x
        # if not, we should pop the first element in the queue and then add new element
        else:
            self.pop()
            self.data.append(d)
    
    # define a pop function here
    # this is achieved by using list's pop(idx) method 
    def pop(self):
        return self.data.pop(0)
    
    # define a item function based on index
    # this is achieved by using list's [] method
    def item(self,idx):
        return self.data[idx]
    
class Monitor(Queue):
    # initialization function of class Monitor and it succeeds Queue class
    # so it requires a parameter called size
    def __init__(self,size):
        super(Monitor,self).__init__()
        # here define a dictionary called M to store three type method
        self.M = {'Regression':'regression','Mean':'mean','Gaussian':'guess'}

    # here define a statucmethod diff
    # there is no self keyword and it can be called by class and instance
    # this function is used to acquire differential values of given data !!!
    # And we set the time to 1 so we just consider difference here
    @staticmethod
    def diff(data):
        # judge if data is a multi-dimensional matrix
        assert isinstance(data,np.ndarray)
        # assign a 0 matrix to dif (data.shape[0]-1*2) two columns represent x and y direction of an image
        # data.shape[0]-1: because we need to use i+1 in the next step to calculate the difference   
        dif = np.zeros((data.shape[0]-1,2))
        for i in range(dif.shape[0]):
            dif[i,0] = data[i+1,0]-data[i,0]
            dif[i,1] = data[i+1,1]-data[i,1]
        return dif
    
    # regression method which requires a 1D data array and then we can use svd to get u, w, vt
    # use these commands to get the regression result 
    @staticmethod
    def regression(data):
        data.reshape(1,-1)
        u,w,vt = linalg.svd(data)
        #-------------------------------???-------------------------- still unclear
        # I think p = np.abs(vt[:,-1])
        p = np.abs(1-vt[-1,:])
        return np.dot(p,data.T)[0]
    
    @staticmethod
    def guess(data):
        # guess is based on Gaussian distribution
        return random.gauss(np.mean(data),np.std(data))
    
    # this function is used to preprocess the data to remove outliers
    @staticmethod
    def preprocess(data):
        mean = np.mean(data)
        std = np.std(data)
        data = data[data>=mean-3*std]
        data = data[data<=mean+3*std]
        return data
    
    # this function is used to update the displacement in x and y diretion according to the values given 
    # t is the time of unit movement and here we set it to 1s 
    @staticmethod
    def update(displacement,t,vel,acc):
        x_dis = displacement[-t,0]+vel[0]*t+0.5*acc[0]*t**2
        y_dis = displacement[-t,1]+vel[1]*t+0.5*acc[1]*t**2
        return np.array([x_dis],[y_dis]])

    # this function is used to process data according to the method chosen
    # all the data should be preprocessed firstly and then use method(data) to process further
    # finally we put the data into our new list and return it
    def process(self,data,method):
        ans = []
        # i and j are both equal to 2 bcs the size of [vel,acc] is 2 and each var's shape is [-1,2] 
        for i in range(2):
            for j in range(2):
                # here we use data[i] to determine we use vel or acc
                # use data[i][:,j] to determine x or y direction in vel and acc
                tmp = self.preprocess(data[i][:,j])
                eval('ans.append(self.'method+'(tmp)')
        return ans

    # this function is used to predict 
    def predict(self,kernelSize,method):
        method = self.M[method]
        assert method
        # make sure the data we handled is np.array type
        data = np.array(self.data)
        # displacement is the data's 2D type
        displacement = data.reshape(-1,2)
        # call self.diff() to acquire vel and acc
        vel = self.diff(displacement)
        acc = self.diff(vel)
        # finally we can update our displacement
        # [:2] means the list[0:2] the first two element in the list
        # [2:] means the list[2:end] the last two element in the list
        return self.update(displacement,1,self.process([vel,acc],method)[:2],self.process([vel,acc],method)[2:])

    



