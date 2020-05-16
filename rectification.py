import cv2
import os
import sys
import glob
import copy
import math
import numpy as np
from numpy import array

# read camera info from txt file
with open('cameraInfo.txt','r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
dist1 = cameraInfo['dist1']
mtx2 = cameraInfo['mtx2']
dist2 = cameraInfo['dist2']
R = cameraInfo['R']
T = cameraInfo['T']

def rectify(img, mtx, dist, root):
    map1, map2 = cv2.initUndistortRectifyMap(mtx,dist,r,p,(img.shape[:2])[::-1],cv2.CV_32FC1)
    im = cv2.remap(img,map1,map2,cv2.INTER_LINEAR)
    if not os.path.exists(root):
        os.makedirs(root)
    cv2.imwrite(root+str(i)+'.png',im)

left = sorted(glob.glob('Undistorted/withOcc/left/*.png'))
root = 'Rectified/withOcc/'
for i in range(len(left)):
    iml = cv2.imread('Undistorted/withOcc/left/'+str(i)+'.png')
    imr = cv2.imread('Undistorted/withOcc/right/'+str(i)+'.png')
    R1, R2, P1, P2, Q, roi1 ,roi2  = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, 
    (iml.shape[:2])[::-1], R, T)
    r, p = R1, P1
    x,y,w,h = roi1
    iml = iml[y:y+h,x:x+w]
    rectify(iml, mtx1, dist1, root+'left/')
    r, p = R2, P2
    x,y,w,h = roi2
    imr = imr[y:y+h,x:x+w]
    rectify(imr, mtx2, dist2, root+'right/')


