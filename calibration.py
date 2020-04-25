import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import copy
import glob
import os

# chessboard size
nb_horizontal = 9
nb_vertical = 6
# world coordinates, x->y->z
# set x and y coordinates
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
# set z coordinates
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# firstly read images to calibrate
left = sorted(glob.glob("left_chess/left*.png"))
right = sorted(glob.glob("right_chess/right*.png"))
imgPoints_l = []
imgPoints_r = []
img_left = []
img_right = []
objpoints = []
for i in range(len(left)):
    iml = cv2.imread(left[i])
    imr = cv2.imread(right[i])
    retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
    imgl = copy.deepcopy(iml)
    imgr = copy.deepcopy(imr)
    cv2.drawChessboardCorners(imgl, (nb_vertical,nb_horizontal), corners_l, retl)
    cv2.drawChessboardCorners(imgr, (nb_vertical,nb_horizontal), corners_r, retr)
    # when and only when both of the left and right images in a pair has the corners
    if retl and retr: 
        imgPoints_l.append(corners_l)
        imgPoints_r.append(corners_r)
        img_left.append(iml)
        img_right.append(imr)
        objpoints.append(objp)
    #cv2.imshow('left', imgl)
    #cv2.waitKey(5)
    #cv2.imshow('right', imgr)
    #cv2.waitKey(10)
#cv2.destroyAllWindows()

# determine the camera matirx 1
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints, imgPoints_l, (imgl.shape[:2])[::-1], None, None)
# determine the camera matirx 2
_, mtx2, dist2,_ ,_ = cv2.calibrateCamera(objpoints, imgPoints_r, (imgr.shape[:2])[::-1], None, None)

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
def undistort_liu(img, mtx, dist, root):
    counter = 0
    if not os.path.exists(root):
        os.makedirs(root)
    for im in img:
        im = cv2.imread(im)
        h,w = im.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(im,mtx,dist,None,newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(root+'/'+str(counter)+'.png', dst)
        counter = counter+1
'''

left = sorted(glob.glob("raw_img/withOcc/left/*.png"))
right = sorted(glob.glob("raw_img/withOcc/right/*.png"))
root = 'UndistortedImgOcc/'
undistort(left, mtx1, dist1, root+'left')
undistort(right, mtx2, dist2, root+'right') 
'''
left = sorted(glob.glob("left/*.png"))
right = sorted(glob.glob("right/*.png"))
root = 'UndistortedImg/'
undistort_liu(left, mtx1, dist1, root+'left')
undistort_liu(right, mtx2, dist2, root+'right') 
'''
# undistort chessboard

left2 = sorted(glob.glob("left_chess/left*.png"))
right2 = sorted(glob.glob("right_chess/right*.png"))
root2 = 'UndistortedChessBoard/'
undistort(left2, mtx1, dist1, root2+'left')
undistort(right2, mtx2, dist2, root2+'right') 

## calibrate via the undistorted chessboard
left_undistort = sorted(glob.glob("UndistortedChessBoard/left/*.png"))
right_undistort = sorted(glob.glob("UndistortedChessBoard/right/*.png"))
imgPoints_left = []
imgPoints_right = []
objpoints_undistort = []
for i in range(len(left_undistort)):
    iml = cv2.imread(left_undistort[i])
    imr = cv2.imread(right_undistort[i])
    retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
    imgl = copy.deepcopy(iml)
    imgr = copy.deepcopy(imr)
    cv2.drawChessboardCorners(imgl, (nb_vertical,nb_horizontal), corners_l, retl)
    cv2.drawChessboardCorners(imgr, (nb_vertical,nb_horizontal), corners_r, retr)
    # when and only when both of the left and right images in a pair has the corners
    if retl and retr: 
        imgPoints_left.append(corners_l)
        imgPoints_right.append(corners_r)
        objpoints_undistort.append(objp)
    #cv2.imshow('left', imgl)
    #cv2.waitKey(5)
    #cv2.imshow('right', imgr)
    #cv2.waitKey(10)
#cv2.destroyAllWindows()

# determine the camera matirx 1
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints_undistort, imgPoints_left, (imgl.shape[:2])[::-1], None, None)
# determine the camera matirx 2
_, mtx2, dist2,_ ,_ = cv2.calibrateCamera(objpoints_undistort, imgPoints_right, (imgr.shape[:2])[::-1], None, None)


# stereo calibration
_, mtx1, dist1, mtx2, dist2, R, T,_ ,_ = cv2.stereoCalibrate(objpoints_undistort, imgPoints_left, imgPoints_right, mtx1, dist1, mtx2, dist2, (imgl.shape[:2])[::-1], flags=cv2.CALIB_FIX_INTRINSIC)


# camera information
cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2, 'R':R, 'T':T}
with open('cameraInfo_Occ.txt','w') as f:
    f.write(str(cameraInfo))