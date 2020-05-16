import cv2
import numpy as np
import glob
import os
import imutils
from tracker import BSTracker


images = glob.glob('../Rectified/withoutOcc/left/*.png')
images.sort(key=lambda x: os.path.getctime(x))
assert images

tracker = BSTracker(80)


for i in range(len(images)):
    img = cv2.imread(images[i])
    text = 'Undetected'
    if not tracker.train(img):
        continue
    
    ret, roi, center, rearest, mask = tracker.detect(img)

    if ret:
        cv2.rectangle(img,(roi[0],roi[1]),(roi[2],roi[3]),(0,255,0),3)
        cv2.circle(img,(int(center[0]),int(center[1])),2, (0, 255, 0),3)
        text = "detected"
        
    cv2.putText(img, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)       
    cv2.imshow('conveyor_without_occlusions',img)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break


