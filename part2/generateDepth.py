import cv2
import numpy as np
import glob
import os
import imutils
from tracker import BSTracker
from numpy import array

images_left = glob.glob('../Rectified/withoutOcc/left/*.png')
images_left.sort(key=lambda x: os.path.getctime(x))
assert images_left
images_right = glob.glob('../Rectified/withoutOcc/right/*.png')
images_right.sort(key=lambda x: os.path.getctime(x))
assert images_right
root = '../DepthMaps/'

for i in range(len(images_left)):
    iml = cv2.imread(images_left[i])
    imr = cv2.imread(images_right[i])
    iml_size = iml.shape[:2][::-1]
    imr = cv2.resize(imr,iml_size)
    gray_l = cv2.cvtColor(iml,cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(imr,cv2.COLOR_BGR2GRAY)
    h,w,c = iml.shape
    winSize = 5
    numberOfDisparities = int((w / 8) + 15) & -16
    stereo = cv2.StereoSGBM_create(0,16,3)
    stereo.setPreFilterCap(32)
    stereo.setP1(8*c*winSize**2)
    stereo.setP2(32*c*winSize**2)
    stereo.setMinDisparity(0)
    stereo.setBlockSize(winSize)
    stereo.setNumDisparities(numberOfDisparities)
    stereo.setDisp12MaxDiff(100)
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleRange(32)
    stereo.setSpeckleWindowSize(0)
    disp = stereo.compute(iml, imr).astype(np.float32) / 16.0
    disp = cv2.medianBlur(disp, 5)
    disp = numberOfDisparities+disp
    depth = disp.copy()
    disp[disp<0] = 0
    w,h = disp.shape
    # make up the "holes" in the map
    # use integration map to accelerate the calculation
    inte = cv2.integral2(disp)[0]
    for m in range(w):
        for n in range(h):
            size = 30
            if not disp[m,n]:
                idx1 = max([m-size,0])
                idx2 = min([w,m+size])
                idx3 = max([n-size,0])
                idx4 = min([h,n+size])
                arr = disp[idx1:idx2,idx3:idx4]
                if np.sum(arr>0):
                    num = np.sum(arr>0)
                else:
                    num = np.inf
                depth[m,n] = (inte[idx1,idx3] + inte[idx2,idx4]-inte[idx2,idx3]-inte[idx1,idx4])/num
    disp = depth
    depth[depth==0] = -1
    # the focal length is 700 and the baseline length is 3.5
    depth = 702.12282068*np.sqrt((-0.11974927)**2+(-0.00019368)**2+(-0.00351544)**2)/depth
    # the minimum depth should be 0, in line with the basic knowledge 
    depth[depth<0] = 0

    if not os.path.exists(root):
        os.makedirs(root)
    cv2.imwrite(root+str(i)+'.png',depth)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break
          

cv2.destroyAllWindows()

