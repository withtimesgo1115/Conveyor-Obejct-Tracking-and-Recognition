import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from utils import opticalPyramid, locateObj, DataLoader, Queue
import time
import glob

## Parameters
path= '../RectifiedImgOcc/right/'
#d = DataLoader.DataLoader(path, size=(1024, 729), cvt=cv2.COLOR_BGR2RGB)
roi = None
his_size = 10
mon_size = 15
minScore = 1500
kernelSize = (200, 200)
stride = (10, 10)
color = [(0, 255, 0), (0, 0, 255)]
##

# create object of History and Path in Queue class
history = Queue.Hist(his_size, kernelSize[0])
monitor = Queue.Path(mon_size, 5)
# This is to acquire current time
t0 = time.time()
# occlusion flag which means flag will change to true if there is some occlusion 
flag = False 
roi = None
right_imgs = sorted(glob.glob(path+'*.png'))
IMG_NUMS = len(right_imgs)
assert IMG_NUMS
f'Reading error, please check the folder!'
print('There are %d pictures in the folder'%(IMG_NUMS))

try:
    img = [cv2.resize(cv2.imread(path+str(i)+'.png'),(1024,729)) for i in range(IMG_NUMS)]
except:
    print('Out of memory. Try to read the images in batches.')
    dir_file = sorted(glob.glob(path+'*.png'))

# handle all frames in the loop. We start from 320 and stop at the last second one bcs we read two frames each time 
for i in range(320, len(img)-1):
    # read two frames 
    im1 = img[i]
    im2 = img[i+1]
    # calculate the score, roi and center based on our function locateObj.cov
    # this function is implemented by using dense optical flow and details can be searched in locateObj.py
    score, roi, center = locateObj.conv(im1, im2, kernelSize, stride, minScore=minScore)
    # im1.copy() can return a copy of image im1.  
    im = im1.copy()

    # update monitor
    monitor.update(history.isFound)
    if max(roi)!=0:
        cv2.rectangle(im, (roi[1], roi[0]), (roi[3], roi[2]), color[history.isFound], 5)
        history.push(center[0])
        monitor.push(center)
        if flag:
            flag = False
    else:
        history.data = []
        # detect the occlusion successfully
        if monitor.detect():
            flag = True
        if flag:                
            x,y = monitor.predict(kernelSize)
            monitor.push([x,y])
            cv2.rectangle(im, (int(y-kernelSize[1]/2), int(x-kernelSize[0]/2)), 
            (int(y+kernelSize[1]/2), int(x+kernelSize[0]/2)), (0, 255, 255), 5)
             
    cv2.imshow('Object Tracking', im)
    cv2.waitKey(1)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break  
cv2.destroyAllWindows() 



#t = time.time()-t0

#fps = t/(i+1)
#print('The implementation Rate is %0.2f fps'%fps)