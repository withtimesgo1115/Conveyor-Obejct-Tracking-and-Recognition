import cv2
import numpy as np
import imutils

class BSTracker(object):
    # constructor
    def __init__(self,his):
        self.history = his
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        self.bs.setHistory(self.history)
        self.trainTimes = 0
    
    # define a train function to read former images
    def train(self,img):
        self.bs.apply(img)
        self.trainTimes += 1
        if self.trainTimes > self.history:
            return 1
        else:
            return 0

    def detect(self,img):
        fg_mask = self.bs.apply(img)
        mask = cv2.dilate(fg_mask, None, iterations=10)
        mask = cv2.erode(mask, None, iterations=10)
        cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        maxArea = 0
        ret = 0
        target = 0
        roi = []
        center = []
        rearest = []
        for c in cnts:
            if cv2.contourArea(c) > maxArea and cv2.contourArea(c) > 2000:
                maxArea = cv2.contourArea(c)
                target = c
                ret = True
        if ret:
            (x,y,w,h) = cv2.boundingRect(target)
            roi = (x,y,x+w,y+h)
            center = np.array([x+w/2,y+h/2])

            maxx = -1
            for point in target:
                if point[0][0]>maxx:
                    maxx = point[0][0]
                    rearest = point[0]

        return ret, roi, center, rearest, mask 

