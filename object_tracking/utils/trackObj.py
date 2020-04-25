import cv2
import numpy as np
import time
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import Kalman, Datastruc, locateObj_lin
from preprocessing import DataLoader

'''
                            Note: 4 predefined STATUS
             Movement Detected?   Previous Movement Successive? BinaryCode  StatusNumber
background          No                          No                (0,0)          0
potentialObj        Yes                         No                (1,0)          1
occlusion           No                          Yes               (0,1)          2
target              Yes                         Yes               (1,1)          3
'''
'''
                                                MONITOR
The monitor in Tracker saves the observed centers of the object in serval previous frames. When the status becomes
2:occlusion, the monitor will predict the path of the object based on the data it collected before. One should set the
prediction method. Now there are 3 methods predefined. It's clear that we shall not work derectly on the center coordinates, 
since the object is dynamic. The outputs of 'prediction' are the predicted new velocity and acceleration of the object. 
Keyword 'Regression' means doing linear regression on the previous object centers to get a new one. 'Mean' is to get the 
mean value of the data. 'Gaussian' is much more funny. It assumes that the speed and acceleration of the object is subject to
Gaussian distribution, for which when the object is under occulusion, the monitor will guess the new speed randomly.
'''
'''
                                        PARAMETERS in Tracker
- d:(DataLoader) instance that reads and preprocesses images
- filter:(Kalman) kalman filter, to denoise and stablize the movement of the object 
- center: (1x2 TUPLE) the observed or predicted center of the object in current frame
- fps: (FLOAT) Frame Per Second, a metric to measure the speed of the algorithm
- objCounter:(INT) a counter to count the number of the previous successive movement(both target and potential object) 
- tarCounter:(INT) a counter to count the number of the previous successive target movement 
- thresh:(1x2 TUPLE) the minimum number of the previous successive movement and previous successive target movement 
- rec:(BOOL) whether activate the classifier
'''

class Tracker(object):
    def __init__(self, path, size, cvt, monsize, thresh, rec=False):
        super(Tracker, self).__init__()
        self.d = DataLoader.DataLoader(path, size=size, cvt=cvt)
        self.filter = Kalman.Kalman()
        self.monitor = Datastruc.Monitor(monsize)
        self.center = np.array([[0],[0]])
        self.fps = 0
        self.objCounter = 0
        self.tarCounter = 0
        self.thresh = thresh
        self.rec = rec
        self.isOcc = False

    def update(self, center, torr):
        '''
        update status of the frame
        see "Note: 4 predefined STATUS" for detail
        '''
        # If nothing detected, center should be (0, 0)
        isDetected = max(center)
        # If counter is larger than threshold, successive condition is met
        objSucc, self.objCounter = self.judge(self.objCounter, self.thresh[0])
        tarSucc, self.tarCounter = self.judge(self.tarCounter, self.thresh[1])
        if isDetected:
            if objSucc:
                # status 3
                if np.abs(center-self.center).max()<=torr:
                    self.tarCounter+=1
                    self.track()
                    return 3
            else:
                # status 1
                # when and only when the object moves small enough can it be considered continuous
                if np.abs(center-self.center).max()<=torr:
                    self.objCounter+=1
                    self.wait()
                    self.isOcc = False
                return 1
        elif tarSucc or self.isOcc:
            # status 2
            self.predict()
            self.isOcc = True
            return 2
        else:
            # status 0
            self.ignore()
            return 0

    def judge(self, counter, thresh):
        isSucc = counter>=thresh
        if not max(self.center):
            counter = 0
        return isSucc, counter

    @abstractmethod
    def ignore(self):
        '''
        status 0
        '''
        pass

    @abstractmethod
    def wait(self):
        '''
        status 1
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        status 2
        '''
        pass

    @abstractmethod
    def track(self):
        '''
        status 3
        '''
        pass
    
    @abstractmethod
    def classify(self):
        '''
        left for classification
        '''
        pass

    @abstractmethod
    def loop(self, idx1, idx2, Filter=False):
        '''
        process images in a loop
        '''
        pass
    
'''
                                        PARAMETERS in DenseTracker
- d:(DataLoader) instance that reads and preprocesses images
- filter:(Kalman) kalman filter, to denoise and stablize the movement of the object 
- center: (1x2 TUPLE) the observed or predicted center of the object in current frame
- fps: (FLOAT) Frame Per Second, a metric to measure the speed of the algorithm
- objCounter:(INT) a counter to count the number of the previous successive movement(both target and potential object) 
- tarCounter:(INT) a counter to count the number of the previous successive target movement 
- thresh:(1x2 TUPLE) the minimum number of the previous successive movement and previous successive target movement 
- rec:(BOOL) whether activate the classifier
- roi:(None or roi) Region of interest
- index:(1x2 TUPLE) kernel index number
- kernel:(1x2 TUPLE) kernel(roi) size
- stride:(1x2 TUPLE) stride length
- sampling:(BOOL) if sampling, step>1. The video will abandon some frames to accelerate the whole process.  
'''
class DenseTracker(Tracker):
    def __init__(self, path, size, cvt, monsize, thresh, kernel, stride, sampling=False, rec=False):
        super(DenseTracker, self).__init__(path, size, cvt, monsize, thresh, rec)
        self.roi = None
        self.index = None
        self.kernel = kernel
        self.stride = stride
        self.step = 3 if sampling else 1
        self.im = [] # current frame
    
    def draw(self, color):
        # draw a rectangle
        cv2.rectangle(self.im,(int(self.center[1]-self.kernel[1]/2), int(self.center[0]-self.kernel[0]/2)), 
        (int(self.center[1]+self.kernel[1]/2), int(self.center[0]+self.kernel[0]/2)), color, 5)
        cv2.imshow('Object Tracking', self.im)
        cv2.waitKey(1) 

    def ignore(self):
        self.filter.__init__()
        cv2.imshow('Object Tracking', self.im)
        cv2.waitKey(1) 
    
    def wait(self):
        # draw a green box
        self.draw((0, 255, 0))

    def predict(self):
        self.center = self.monitor.predict(self.kernel, 'Mean')
        # draw a yellow box
        self.draw((0, 255, 255))

    def track(self):
        # draw a red box
        self.draw((0, 0, 255))

    def loop(self, idx1, idx2, Filter=False):
        t0 = time.time()
        if not idx2:
            # idx2=0 means that read all the files in the folder
            idx2 = self.d.len-1
        index = self.index
        roi = self.roi
        for i in range(idx1, idx2, self.step):
            im1 = self.d.getItem(i)
            im2 = self.d.getItem(i+1)
            self.im = im2
            roi, center, index = locateObj_lin.conv(im1, im2, self.kernel, self.stride, index=index)
            if Filter:
                center = self.filter.filt(center)
            status = self.update(center, min(self.stride))
            self.center = center if max(center) else self.center
            self.monitor.push(self.center)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break 
        cv2.destroyAllWindows()
        t = time.time()-t0
        self.fps = (idx2-idx1)/(t*self.step)
        print('The implementation Rate is %0.2f fps'%self.fps)

class SparseTracker(DenseTracker):
    def __init__(self, path, size, cvt, monsize, thresh, kernel, stride, featnumber, sampling=False, rec=False):
        super().__init__(path, size, cvt, monsize, thresh, kernel, stride, sampling=sampling, rec=rec)
        self.fn = featnumber

    def loop(self, idx1, idx2, Filter=False):
        t0 = time.time()
        if not idx2:
            # idx2=0 means that read all the files in the folder
            idx2 = self.d.len-1
        index = self.index
        roi = self.roi
        feat = None
        center = np.array([[0],[0]])
        for i in range(idx1, idx2, self.step):
            im1 = self.d.getItem(i)
            im2 = self.d.getItem(i+1)
            self.im = im2
            status = self.update(center, min(self.stride))
            print(status)
            if status is 3:
                feat,center,roi = locateObj_lin.track(im1, im2, roi, self.fn, feat, self.kernel)
            else:
                roi, center, index = locateObj_lin.conv(im1, im2, self.kernel, self.stride, index=index)
            if Filter:
                center = self.filter.filt(center)
            self.center = center if max(center) else self.center
            self.monitor.push(self.center)
        cv2.destroyAllWindows()
        t = time.time()-t0
        self.fps = (idx2-idx1)/(t*self.step)
        print('The implementation Rate is %0.2f fps'%self.fps)

class SubTracker(object):
    def __init__(self):
        super(SubTracker, self).__init__()