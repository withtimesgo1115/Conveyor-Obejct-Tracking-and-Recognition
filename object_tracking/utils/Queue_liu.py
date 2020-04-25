import numpy as np
from numpy import linalg

class Queue_liu():
    def __init__(self,size):
        super(Queue,self).__init__()
        self.size = size
        self.data = []

    def push(self,d):
        if len(self.data)<self.size:
            self.data.append(d)
        else:
            self.pop()
            self.data.append(d)

    def pop(self):
        return self.data.pop(0) or []
    
    def item(self,idx):
        return self.data[idx]

class Hist(Queue):
    def __init__(self,size,torr):
        super(Hist,self).__init__(size)
        self.torr = torr
        
    @property
    def isFound(self):
        isFound = False
        if len(self.data)>=self.size:
            if (max(self.data)-min(self.data)) < self.torr:
                isFound = True
            else:
                isFound = False
        return isFound

def regression(data):
    data = data.reshape(1,-1)
    u, w, v = linalg.svd(data)
    p = np.abs(1-v[-1,:])
    return np.dot(p,data.T)[0]

def preprocess(data):
    mean = np.mean(data)
    std = np.std(data)
    data = data[data>=mean-3*std]
    data = data[data<=mean+3*std]
    return data

class Path(Queue):
    def __init__(self,size):
        super(Path,self).__init__(size)
    
    def predict(self,kernelSize):
        data = np.array(self.data)
        displacement = data.reshape(-1,2)
        vel = zeros((displacement.shape[0]-1,2))
        for i in range(vel.shape[0]):
            vel[i,0] = displacement[i+1,0]-displacement[i,0]
            vel[i:1] = displacement[i+1,1]-displacement[i,1]
        acc = np.zeros((vel.shape[0]-1,2))
        for i in range(acc.shape[0]):
            acc[i,0] = vel[i+1,0]-vel[i,0]
            acc[i,1] = vel[i+1,1]-vel[i,1]
        t = 1
        x_vel = preprocess(vel[:,0])
        y_vel = preprocess(vel[:,1])
        x_acc = preprocess(acc[:,0])
        y_acc = preprocess(acc[:,1])
        
        x_vel = np.mean(x_vel)
        y_vel = np.mean(y_vel)
        x_acc = np.mean(x_acc)
        y_acc = np.mean(y_acc)
        x_dis = displacement[-t,0]+x_vel*t+0.5*x_acc*t**2
        y_dis = displacement[-t,1]+y_vel*t+0.5*y_acc*t**2
        return np.array([[x_dis], [y_dis]])







        



