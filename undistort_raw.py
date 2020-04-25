import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import copy
import glob
from numpy import array
import os


# firstly read images to calibrate
with open('cameraInfo.txt','r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
mtx2 = cameraInfo['mtx2']
dist1 = cameraInfo['dist1']
dist2 = cameraInfo['dist2']
R = cameraInfo['R']
T = cameraInfo['T']

# undistortion
def undistort(img, mtx, dist, root):
    counter = 0
    if not os.path.exists(root):
        os.makedirs(root)
    for im in img:
        im = cv2.imread(im)
        dst = cv2.undistort(im, mtx, dist)
        cv2.imwrite(root+'/'+str(counter)+'.png', dst)
        counter = counter+1
'''
left = glob.glob("raw_data/Stereo_conveyor_with_occlusions/left/*.png")
right = glob.glob("raw_data/Stereo_conveyor_with_occlusions/right/*.png")
root = 'raw_data/UndistortedImgOcc/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 
'''

left = sorted(glob.glob("left/*.png"))
right = sorted(glob.glob("right/*.png"))
root = 'UndistortedImg/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 