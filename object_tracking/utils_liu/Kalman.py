import cv2
import numpy as np
import math 
import random

class Kalman(object):
    def __init__(self):
        super(Kalman).__init__()
        # the initial state (6x1)
        # it contains displacement, velocity and accelaration
        # including x, v_x, a_x, y, v_y, a_y
        self.X = np.zeros((6,1))
        # the initial uncertainty (6x6)
        # let's assume a large initial value since at the very first the uncertainty is high
        uc = 1
        self.P = = uc*np.eye(6)
        # the external motion (6x1)
        self.u = np.zeros((6,1))
        # the transition matrix (6x6)
        # ??? how can you set these values in this way? ----------------still unclear----------------------- 
        self.F = np.array([1,1,0.5,0,0,0],
                            [0,1,1,0,0,0],
                            [0,0,1,0,0,0],
                            [0,0,0,1,1,0.5],
                            [0,0,0,0,1,1],
                            [0,0,0,0,0,1])
        # the observation matrix (2x6) 
        # in this problem, we only want to know the pose so we set H[0][0] and H[1][3] = 1
        self.H = np.array([[1,0,0,0,0,0],[0,0,0,1,0,0]])
        # the measurement uncertainty (2x1)
        self.R = np.array([[1],[1]])
        # the unit matrix
        self.I = np.eye(6)

    # this is update calculation of Kalman filter and it requires measurement value Z
    def update(self,Z):
        y = Z-np.dot(self.H,self.X)
        S = np.dot(np.dot(self.H,self.P),np.transpose(self.H))+self.R
        K = np.dot(np.dot(self.P,np.transpose(self.H)),np.linalg.pinv(S))
        self.X = self.X + np.dot(K,y)
        self.P = np.dot(self.I-np.dot(self.K,self.H),self.P)
    
    # this is predict calculation of Kalman filter and nothing is required 
    def predict(self):
        self.X = np.dot(self.X,self.F)+self.u
        self.P = np.dot(np.dot(self.F,self.P),np.transpose(self.F))
    
    # Kalman filter's work process
    # At first, call update() and give the value to pose
    # then use these new correct pose data to predict
    # in the end, return pose  
    def filt(self,Z):
        self.update(Z)
        # [[x],[y]]
        pose = np.array([[self.X[0].T],[self.X[3].T]])
        self.predict()
        return pose
