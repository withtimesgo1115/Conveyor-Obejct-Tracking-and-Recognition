import cv2
import numpy as np
import glob
import os
import imutils
from tracker import BSTracker
from numpy import array

# read camera info from txt file
with open('../cameraInfo.txt','r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
dist1 = cameraInfo['dist1']
mtx2 = cameraInfo['mtx2']
dist2 = cameraInfo['dist2']
R = cameraInfo['R']
T = cameraInfo['T']


images_left = glob.glob('../Rectified/withoutOcc/left/*.png')
images_left.sort(key=lambda x: os.path.getctime(x))
assert images_left
images_right = glob.glob('../Rectified/withoutOcc/right/*.png')
images_right.sort(key=lambda x: os.path.getctime(x))
assert images_right

tracker1 = BSTracker(80)
tracker2 = BSTracker(80)
#baseline = int(-T[0])
baseline = 0.12

def extract_keypoints_surf(img1, img2):
    """
    use surf to detect keypoint features
    remember to include a Lowes ratio test
    input: img1, img2: the previous and current image
    K: camera matrix
    baseline: the length of baseline
    we use img1 and img2 to reconstruct 3D object points, then use 2D image
    points in img1(reference image) to do PnP. 
    """
    # So first extract surf features
    surf = cv2.xfeatures2d_SURF.create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    # Flann is more stable
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    match_points1 = []
    match_points2 = []
    for m in matches:
        match_points1.append(kp1[m.queryIdx].pt)
        match_points2.append(kp2[m.trainIdx].pt)
    p1 = np.array(match_points1).astype(np.float32)
    p2 = np.array(match_points2).astype(np.float32)

    return p1, p2


def triangulate(p1,p2,mtx1,mtx2,baseline):
    ##### ############# ##########
    ##### Do Triangulation #######
    ##### ########################
    #project the feature points to 3D with triangulation
    
    #projection matrix for Left and Right Image
    M_left = mtx1.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = mtx2.dot(np.hstack((np.eye(3), np.array([[-baseline, 0, 0]]).T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]

    return land_points.T, p1


def track3D(img1,img2,roi1,roi2,center1,center2,mtx1,mtx2,baseline):
    imgCropped1 = img1[roi1[1]:roi1[3],roi1[0]:roi1[2]]
    imgCropped2 = img2[roi1[1]:roi2[3],roi2[0]:roi2[2]]
    kp_roi1, kp_roi2 = extract_keypoints_surf(imgCropped1,imgCropped2)

    if len(kp_roi1)!=0 and len(kp_roi2)!=0:
        kp1=[]
        kp2=[]
        for i in range(len(kp_roi1)):
            p1 = (int(kp_roi1[i][0]),int(kp_roi1[i][1]))
            p2 = (int(kp_roi2[i][0]),int(kp_roi2[i][1]))
            kp_img1 = (p1[0]+x1, p1[1]+y1)
            kp_img2 = (p2[0]+x2, p2[1]+y2)
            kp1.append(kp_img1)
            kp2.append(kp_img2)
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        P, _ = triangulate(kp1 , kp1 , mtx1, mtx2, baseline)
        point3D = np.mean(P,axis = 0)
    else:
        P, _ = triangulate(center1 , center2 , mtx1, mtx2, baseline)
        point3D = P[0]

    return img1, img2, point3D




for f1, f2 in zip(images_left,images_right):
    im1 = cv2.imread(f1)
    im2 = cv2.imread(f2)
    im1_size = im1.shape[:2][::-1]
    im2 = cv2.resize(im2,im1_size)
    text = 'Undetected'

    if not (tracker1.train(im1) and tracker2.train(im2)):
        continue
    
    ret1, roi1, center1, rearest1, mask1 = tracker1.detect(im1)
    ret2, roi2, center2, rearest2, mask2 = tracker2.detect(im2)

    if ret1 and ret2:
        text = 'Detected'
        cv2.rectangle(im1, (roi1[0],roi1[1]), (roi1[2],roi1[3]), (0, 255, 0), 2)
        cv2.circle(im1,(int(center1[0]),int(center1[1])),2,(0, 255, 0),3)
        #cv2.rectangle(im2, (roi1[0],roi1[2]), (roi1[1],roi1[3]), (0, 255, 0), 2)
        P, _ = triangulate(center1 , center2 , mtx1, mtx2, baseline)
        point3D = P[0]
        text2 = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0]), float(point3D[1]),float(point3D[2]))
    
    cv2.putText(im1, text2, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, 8)
    cv2.putText(im1, "Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    cv2.imshow('left img',im1)
    cv2.imshow('left mask',mask1)
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break
          

cv2.destroyAllWindows()



 