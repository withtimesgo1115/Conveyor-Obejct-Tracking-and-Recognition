import numpy as np
from numpy import linalg
#import random

class Queue():
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
        return self.data.pop(0) or []

    def item(self, idx):
        return self.data[idx]

class Hist(Queue):
    def __init__(self, size, torr):
        super(Hist, self).__init__(size)
        self.torr = torr
    
    @property
    def isFound(self):
        isFound = False
        if len(self.data)>=self.size:
            if (max(self.data)-min(self.data))<self.torr:
                isFound = True
            else:
                isFound = False 
        #if isFound:
        #    print('object detected')
        return isFound

def regression(data):
    u, w, v = linalg.svd(data.T)
    p = abs(1-v[:,-1])
    p = p/sum(p)
    d = np.dot(data.T, p)
    return d
    
class Path(Queue):
    def __init__(self, size, torr):
        super(Path, self).__init__(size)
        self.counter = 0
        self.torr = torr

    def detect(self):
        # if and only if find object in last frame and nothing in this frame
        if self.counter>=self.torr:
            print('Lose target!')
            return True
        else:
            return False

    def predict(self, kernelSize):
        data = np.array(self.data)
        print(data)
        x_shift = np.zeros((data.shape[0]-1, 1))
        y_shift = np.zeros_like(x_shift)
        for i in range(data.shape[0]-1):
            x_shift[i] = data[i+1, 0] - data[i, 0]
            y_shift[i] = data[i+1, 1] - data[i, 1] 
        x_s = regression(x_shift)[0]
        y_s = regression(y_shift)[0]
        x_pose = data[-1,0]+x_s
        y_pose = data[-1,1]+y_s
        return x_pose, y_pose

    def update(self, isFound):
        if isFound:
            self.counter+=1
        else:
            self.counter=0