import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import copy
import glob
from numpy import array
import os

with open('cameraInfo_Occ.txt', 'r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
mtx2 = cameraInfo['mtx2']
dist1 = cameraInfo['dist1']
dist2 = cameraInfo['dist2']
R = cameraInfo['R']
T = cameraInfo['T']

# rectification
def rectify(im, mtx, dist, root):
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, r, p, (im.shape[:2])[::-1], cv2.CV_32FC1)
    #map1, map2 = cv2.initUndistortRectifyMap(mtx1, dist1, r, p, (im.shape[:2])[::-1], cv2.CV_32FC1)
    im = cv2.remap(im, map1, map2, cv2.INTER_LINEAR)
    if not os.path.exists(root):
        os.makedirs(root)
    cv2.imwrite(root+'/'+str(i)+'.png',im)


left = glob.glob("UndistortedImgOcc/left/*.png")
root = 'RectifiedImgOcc/'
for i in range(len(left)):
    iml = cv2.imread("UndistortedImgOcc/left/"+str(i)+".png")
    imr = cv2.imread("UndistortedImgOcc/right/"+str(i)+".png")
    R1, R2, P1, P2, Q, roi1 ,roi2  = cv2.stereoRectify(mtx1, dist1, mtx2, dist2, 
    (iml.shape[:2])[::-1], R, T)
    r, p = R1, P1
    x,y,w,h = roi1
    iml = iml[y:y+h,x:x+w]
    rectify(iml, mtx1, dist1, root+'left')
    r, p = R2, P2
    x,y,w,h = roi2
    imr = imr[y:y+h,x:x+w]
    rectify(imr, mtx2, dist2, root+'right')
    #print('o'*i+'x'+'-'*(len(left)-1-i))