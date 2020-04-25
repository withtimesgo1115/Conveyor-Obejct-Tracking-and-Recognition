import cv2
import numpy as np
import os
import sys
import math
import glob
from matplotlib import pyplot as plt


path = 'RectifiedImg_lin/left'
img_list = []
#img_list = os.listdir(path)
#img_list.sort()
#img_list.sort(key = lambda x: int(x[:-4]))
#print(img_list)
images = glob.glob(path+'/*.png')
images.sort(key=lambda x: os.path.getctime(x))
#images = glob.glob(path+'/*.png')
fps = 15


for fname in images: 
    img = cv2.imread(fname) 
    h,w,layers = img.shape
    size = (w,h)
    img_list.append(img)

video = cv2.VideoWriter("VideoTest1.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, size)


for i in range(len(img_list)):
    #img = cv2.imread(item)
    video.write(img_list[i])
video.release()

