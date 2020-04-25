import numpy as np
from numpy import linalg
import random

class Queue(object):
    def __init__(self, size):
        super(Queue, self).__init__()
        self.size = size
        self.data = []
    
    def push(self, d):
        if len(self.data)<self.size:
            self.data.append(d)
        else: 
            self.pop()
            self.data.append(d)

    def pop(self):
        return self.data.pop(0)

    def item(self, idx):
        return self.data[idx]

class Monitor(Queue):
    def __init__(self, size):
        super(Monitor, self).__init__(size)
        self.M = {'Regression':'regression', 'Mean':'mean', 'Gaussian':'guess'}

    @staticmethod
    def diff(data):
        assert isinstance(data, np.ndarray)
        dif = np.zeros((data.shape[0]-1, 2))
        for i in range(dif.shape[0]):
            dif[i,0] = data[i+1, 0] - data[i, 0]
            dif[i,1] = data[i+1, 1] - data[i, 1]
        return dif
    
    @staticmethod
    def regression(data):
        data = data.reshape(1,-1)
        u, w, vt = linalg.svd(data)
        p = np.abs(1-vt[-1,:])
        return np.dot(p, data.T)[0]
    
    @staticmethod
    def guess(data):
        # guess based on Gaussian distribution 
        return random.gauss(np.mean(data), np.std(data))
    
    @staticmethod
    def mean(data):
        return np.mean(data)

    @staticmethod
    def preprocess(data):
        mean = np.mean(data)
        std = np.std(data)
        data = data[data>=mean-3*std]
        data = data[data<=mean+3*std]
        return data
    
    @staticmethod
    def update(displacement, t, vel, acc):
        x_dis = displacement[-t,0]+vel[0]*t+0.5*acc[0]*t**2
        y_dis = displacement[-t,1]+vel[1]*t+0.5*acc[1]*t**2
        return np.array([[x_dis],[y_dis]])

    def process(self,data, method):
        ans = []
        for i in range(2):
            for j in range(2):
                tmp = self.preprocess(data[i][:,j])
                eval('ans.append(self.'+method+'(tmp))')
        return ans
        
    def predict(self, kernelSize, method):
        method = self.M[method]
        assert method
        data = np.array(self.data)
        displacement = data.reshape(-1,2)
        vel = self.diff(displacement)  
        acc = self.diff(vel)
        return self.update(displacement, 1, self.process([vel, acc], method)[:2], self.process([vel, acc], method)[2:])