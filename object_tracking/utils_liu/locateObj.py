import cv2
import numpy as np

def conv(im1,im2,kernelSize,stride,inertia=None,minScore=1500,eroRate=21,
dilRate=25,winsize=15,Euclian=False,horiParam=0.8,minSpeed=1,inertiaParam=0.5,index=None):
    '''
    -----------------------
    Paramters Introduction
    -----------------------
    - kernelSize:(1x2 tuple) the size of ROI. The larger it is, the less time the algorithm takes, with the less accuracy
    - stride: (1x2 tuple) the stride of the kernel. the larger stride leads to a higher searching speed and lower accuracy
    it affects the speed much more than kernel size
    - inertia: (None or ROI) increase the probability that the ROI remain the same position with the last searching result. 
    If it is no need to do that, set it to be None
    - minScore: (int) a searching area with score more than minScore is considered to be a candidate of ROI
    - eroRate: (int better odd number) the kernel size in erosion operation
    - dilRate: (int better odd number) the kernel size in dilation operation
    - winsize: (int better odd number) the window size in dense optical flow
    - Euclian: (bool) if Euclian is True, the displacement of a pixel is defined as the 2-order norm of (x_shift,y_shift)
    which is given by dense optical flow. Otherwise, the displacement if defined as (a*x_shift+(1-a)*y_shift)/2
    - horiParam: (float, 0<&&<1) a in the last line
    - minSpeed: (float, 1 here) if a pixel's displacement is larger than minSpeed, it can be seen as dynamic 
    - inertialParam: (float, better < 0.5) if inertia is not None, inertialParam decides the probability that ROI
    remains the same. the high inertialParam is corresponding to a higher probabilty
    - index: (None or index) to accelarate the searching, input the index found in the last loop. then the function will only
    look for a new ROI around the last ROI
    -------------------
    OUTPUTS
    -------------------
    - roi: (1x4 tuple) the border of ROI (x0,y0,x1,y1)
    - center: (1x2 tuple) the center of ROI (y,x)
    - index: (1x2 tuple) the serial number of the kernel (i,j) i stands for the row and j stands for conlumn
    -------------------
    '''
    r, c = im1.shape[:2]
    gray1, gray2 = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY), cv2.cvtColor(im2,COLOR_RGB2GRAY)
    # create flow map
    opParam = dict(pyr_scale=0.5, levels=5, winsize=winsize, iterations=5, poly_n=1, poly_sigma=0, flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1,gray2,None,**opParam)
    # flow returned by dense optical flow function contains distance values both in x and y direction 
    # what we want here is to acquire scale value in x and y direction separately
    # if we use Euclidian distance, it is the sqrt sum of square value in x and y direction
    # else it is weighted average value of x and y direction's data
    # Due to the larger movement in x direction of objects in this video, we should set weight of x larger 
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1])
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed] = 0
    res = cv2.erode(res,np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res,np.ones((dilRate,dilRate),np.uint8))
    # generate the kernel
    step_r = int(np.ceil((r-kernelSize[0])/stride[0]))
    step_c = int(np.ceil((c-kernelSize[1])/stride[1]))
    score, I = [], []
    if not isinstance(index, tuple):
        # there is no object in the frame currently, search globally 
        for i in range(step_r+1):
            for j in range(step_c+1):
                score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0],j*stride[1]:j*stride[1]+kernelSize[1]]))
                I.append((i,j))
    else:
        # there is an object, we only need to search in the neighbourhood
        # ????? How can you determine the neighbourhood area AND I think this method can decrease the score
        # since we sum the values in a small region------------------------Need help!!!!!--------------------
        for i in range(index[0]-1,index[0]+2):
            for j in range(index[1]-1,index[1]+2):
                score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0],j*stride[1]:j*stride[1]+kernelSize[1]]))
                I.append((i,j))
    if inertia:
        # this is to calculate the inertia score
        iner_score = np.sum(res[inertia[0]:inertia[2],inertia[1]:inertia[3]])*(inertialParam+1)
    else:
        iner_score = None
    # find the ROI with the highest score
    s = max(score)
    idx = score.index(s)
    center = np.array([[0],[0]])
    if iner_score >= s >= minScore:
        roi = inertia
    elif s > minScore:
        roi = (I[idx][0]*stride[0],I[idx][1]*stride[1],I[idx][0]*stride[0]+kernelSize[0],I[idx][1]*stride[1]+kernelSize[1])
    else:
        roi = (0,0,0,0)
        index = None
    #  ------Question-------------
    #  why we still need to find the center of ROI to get the final ROI
    #  we have found the ROI in the last step, right? 
    if max(roi) != 0:
        # np.mgrid is to construct a multi-dimensional data
        # [a,b,c,...] means how many dimensions
        # In our project, we just need to create 2D dimensional data: x and y 
        p = np.mgrid[0:im1.shape[0],0:im1.shape[1]]
        # here we use tuple to assign values to x and y
        x = p[0,:]
        y = p[1,:]
        # the aim of creating a mask is to find the center of ROI we found 
        mask = np.zeros((im1.shape[0],im1.shape[1])) 
        # make moving object area highlight (set to 1 others are 0) 
        mask[roi[0]:roi[2],roi[1]:roi[3]] = res[roi[0]:roi[2],roi[1]:roi[3]]
        # center means the average value so we can calculate the whole scores in x direction and 
        # then devide the highest score s to get the mean value and that result is the x coordinate of the center
        # the same solution for y coordinate
        center[0]=np.sum(mask*x)/s
        center[1]=np.sum(mask*y)/s
        roi = (int(center[0]-kernelSize[0]/2),int(center[1]-kernelSize[1]/2),
        int(center[0]+kernelSize[0]/2),int(center[1]+kernelSize[1]/2))
        index = I[idx]
    return roi, center, index


def track(im1,im2,roi,featnumber,feat,kernelSize,minSpeed=1):
    '''
    Parameters Introduction
    - roi: (1x4 tuple) the pre-defined ROI
    - featnumber: (int) the max number of feature points
    - feat: (None or list) the feature points got from the last loop(used to track)
    - kernelSize: (1x2 tuple) the size of ROI 
    - minSpeed: (float) if a pixel's displacement is larger than minSpeed, it can be seen as dynamic
    Outputs:
    - kp2: (None or list) feature points tracked via sparse optical flow, if there is no points, return None
    - center: (1x2 tuple) center of ROI
    - roi: (1x4 tuple) roi got from sparse optical flow
    '''
    gray1, gray2 = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY), cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)
    # get the ROI_mask area
    ROI_mask = np.zeros(gray1.shape)
    # set this ROI area highlight
    ROI_mask[roi[0]:roi[2],roi[1]:roi[3]] = 1
    # initialize center with np.array([[0],[0]])
    center = np.array([[0],[0]])
    # if feat is not np.ndarray, then extract the keypoints and match them by using sparse optical flow
    if not isinstance(feat,np.ndarray):
        orb = cv2.ORB_create(nfeatures=featnumber)
        kp1,des1 = orb.detectAndCompute(gray1,np.uint8(ROI_mask))
        feat = []
        if len(kp1):
            for i in range(len(kp1)):
                # use feat to store the coordinates of keypoints
                feat.append(kp1[i].pt)
        else:
            return None, center, (int(center[0]-kernelSize[0]/2,int(center[1]-kernelSize[1]/2),
            int(center[0]+kernelSize[0]/2),int(center[1]+kernelSize[1]/2)))
    # transfer feat to np.array and set its datatype to float32
    feat = np.array(feat,dtype=np.float32)
    # use sparse optical flow to calculate feat2
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat, np.uint8(ROI_mask))
    kp2 = np.array([feat2[i,:]  for i in range(feat2.shape[0]) if (status[i] and error[i]>minSpeed)], dtype=np.float32)
    if kp2.shape[0]:
        # ??? why we use x=kp2[:,1] y=kp2[:,0]---------------
        center[0] = np.mean(kp2[:,1])
        center[1] = np.mean(kp2[:,0])
    else:
        kp2 = None
    roi = (int(center[0]-kernelSize[0]/2),int(center[1]-kernelSize[1]/2),
    int(center[0]-kernelSizep[0]/2),int(center[1]+kernelSize[1]/2))
    return kp2, center, roi

    def sptrack(im1,im2,roi,featnumber,feat,kernelSize,minSpeed=1):
        pass
