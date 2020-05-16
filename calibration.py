import cv2
import os
import sys
import glob
import copy
import math
import numpy as np

# checkerboard grid information
nb_horizontal = 9
nb_vertical = 6
# unit is meter
size_per_grid = 0.0336
# prepare an array to store 3D points like (1,1,0) (2,3,0) (5,5,0)
objp = np.zeros((nb_vertical*nb_horizontal,3),np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
objp = objp * size_per_grid
# assign folder path
left = sorted(glob.glob('../left_chess/*.png'))
right = sorted(glob.glob('../right_chess/*.png'))
# prepare list to store points and images
imgPoints_l = []
imgPoints_r = []
img_left = []
img_right = []
objPoints = []

for i in range(len(left)):
    # read img and find corners
    imgl = cv2.imread(left[i])
    imgr = cv2.imread(right[i])
    retl, corners_l = cv2.findChessboardCorners(imgl,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imgl,(nb_vertical,nb_horizontal))
    iml = copy.deepcopy(imgl)
    imr = copy.deepcopy(imgr)
    if retl and retr:
        imgPoints_l.append(corners_l)
        imgPoints_r.append(corners_r)
        img_left.append(iml)
        img_right.append(imr)
        objPoints.append(objp)

# calibrate single camera
_, mtx1, dist1, _, _ = cv2.calibrateCamera(objPoints,imgPoints_l,(iml.shape[:2])[::-1],None,None)
_, mtx2, dist2, _, _ = cv2.calibrateCamera(objPoints,imgPoints_r,(imr.shape[:2])[::-1],None,None) 

# define the undistort function
def undistort(images,mtx,dist,root):
    counter = 0
    if not os.path.exists(root):
        os.makedirs(root)
    for im in images:
        im = cv2.imread(im)
        dst = cv2.undistort(im,mtx,dist)
        cv2.imwrite(root+str(counter)+'.png',dst)
        counter += 1

# get undistorted chessboard images
root = 'Undistorted/chessboard/'
undistort(left,mtx1,dist1,root+'left/')
undistort(right,mtx2,dist2,root+'right/')
# get undistorted dataset without occlusion
root2 = 'Undistorted/withoutOcc/'
left_without = sorted(glob.glob('../left/*.png'))
right_without = sorted(glob.glob('../right/*.png'))
undistort(left_without,mtx1,dist1,root2+'left/')
undistort(right_without,mtx2,dist2,root2+'right/')
# get undistorted dataset with occlusion
root3 = 'Undistorted/withOcc/'
left_with = sorted(glob.glob('../raw_img/withOcc/left/*.png'))
right_with = sorted(glob.glob('../raw_img/withOcc/right/*.png'))
undistort(left_with,mtx1,dist1,root3+'left/')
undistort(right_with,mtx2,dist2,root3+'right/')

# calibrate single camera using undistored dataset
left_undistorted = sorted(glob.glob(root+'left/*.png'))
right_undistorted = sorted(glob.glob(root+'right/*.png'))
imgPoints_l_undistored = []
imgPoints_r_undistored = []
objpoints_undistorted = []
img_left_undistorted = []
img_right_undistorted = []
for i in range(len(left_undistorted)):
    iml = cv2.imread(left_undistorted[i])
    imr = cv2.imread(right_undistorted[i])
    retl, corners_l = cv2.findChessboardCorners(iml,(nb_vertical,nb_horizontal))
    retr, corners_r = cv2.findChessboardCorners(imr,(nb_vertical,nb_horizontal))
    imgl = copy.deepcopy(iml)
    imgr = copy.deepcopy(imr)
    if retl and retr: 
        imgPoints_l_undistored.append(corners_l)
        imgPoints_r_undistored.append(corners_r)
        img_left_undistorted.append(iml)
        img_right_undistorted.append(imr)
        objpoints_undistorted.append(objp)
# determine the camera matirx 1
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints_undistorted, imgPoints_l_undistored, (imgl.shape[:2])[::-1], None, None)
# determine the camera matirx 2
_, mtx2, dist2,_ ,_ = cv2.calibrateCamera(objpoints_undistorted, imgPoints_r_undistored, (imgr.shape[:2])[::-1], None, None)
# stereo calibrate
_, mtx1, dist1, mtx2, dist2, R, T,_ ,_ = cv2.stereoCalibrate(objpoints_undistorted, imgPoints_l_undistored, imgPoints_r_undistored, mtx1, dist1, mtx2, dist2, (imgl.shape[:2])[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
# record camera info
cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2,'R':R, 'T':T}
with open('cameraInfo.txt','w') as f:
    f.write(str(cameraInfo))
