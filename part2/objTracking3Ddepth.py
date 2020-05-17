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
images_depth = glob.glob('../DepthMaps/*.png')
images_depth.sort(key=lambda x: os.path.getctime(x))
assert images_depth

tracker = BSTracker(80)

for i in range(len(images_left)):
    im = cv2.imread(images_left[i])
    depth_img = cv2.imread(images_depth[i])
    print(depth_img)
    text = 'Undetected'
    if not tracker.train(im):
        continue
    
    ret, roi, center, rearest, mask = tracker.detect(im)
    if ret:
        cv2.rectangle(im,(roi[0],roi[1]),(roi[2],roi[3]),(0,255,0),3)
        cv2.circle(im,(int(center[0]),int(center[1])),2, (0, 255, 0),3)
        text = "detected"
        depth = np.mean(depth_img[int(center[1])-5:int(center[1])+5, int(center[0])-5:int(center[0])+5])
        #print(depth)
        text2 = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(center[0]), float(center[1]),float(depth))

    cv2.putText(im, text2, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
    cv2.putText(im, "Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    cv2.imshow('left img',im)
    cv2.imshow('left mask',mask)
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break
          

cv2.destroyAllWindows()