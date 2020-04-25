import cv2
import numpy as np

def conv(im1, im2, kernelSize, stride, inertia=None, minScore=1500, eroRate=21, 
dilRate=25, winsize=15, Euclian=False, horiParam=0.8, minSpeed=1, inertialParam=0.5, index=None):
    '''
    PARAMETERS
    - kernerSize:(1x2TUPLE) the size of ROI. The larger it is, the less time the algorithm takes, with the loss of accuracy. 
    - stride:(1x2TUPLE) the stride of the kernel. Lager stride leads to a higher seaching speed and lower accuracy. It affectes
    the speed much more than the kernel size
    - inertia:(None or roi) increase the probability that the ROI remain the same with the last searching result. If it is of no need, set
    it to be None
    - minScore:(INT) a searching area with score more than minScore is considered to be a candidate of ROI
    - eroRate:(INT, better odd) the kernel size in erosion operation
    - dilate:(INT, better odd) the kernel size in dilation operation
    - winSize:(INT, better odd) the window size in dense optical flow
    - Euclian:(BOOL) if Euclian is set True, the displacement of a pixel is defined as the 2-order norm of (x_shift, y_shift),
    which is given by dense optical flow. Otherwise, the displacement is define as  (a*x_shift+(1-a)*y_shift)/2. 
    - horiParam:(FLOAT, 0<&&<1) a in the last line
    - minSpeed:(FLOAT, 1 here) if a pixel's displacement is larger than minSpeed, it can be seen as dynamic.(According to assumption 1)
    - inertialParam:(FLOAT, better <0.5) if inertia is not None, inertialParam decides the probability that the ROI remains the same. The high
    inertialParam is corresponding with a higher probability
    - index(None or index): to accelerate the searching, input the index found in the last loop. Then the function will only look for a new
    ROI around the last ROI
    OUTPUTS
    - roi:(1x4TUPLE) the border of ROI, (x0, y0, x1, y1)
    - center(1x2TUPLE) the center of ROI, (y, x)
    - index(1x2TUPLE) the serial number of the kernel (i, j), i stands for the row and j stand for the column
    '''
    r, c = im1.shape[:2]
    gray1, gray2 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    # create flow map
    opParam = dict(pyr_scale=0.5, levels=5, winsize=winsize, iterations=5, poly_n=1, poly_sigma=0, flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **opParam)
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1])
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed]= 0
    res = cv2.erode(res, np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res, np.ones((dilRate,dilRate),np.uint8))
    # see the flow map here
    #cv2.imshow('thresh', res*255)
    #cv2.waitKey(1)
    # generate the kernel 
    step_r = int(np.ceil((r-kernelSize[0])/stride[0]))
    step_c = int(np.ceil((c-kernelSize[1])/stride[1]))
    score, I = [], []
    if not isinstance(index, tuple):
        ## there is no object in the frame currently, search globally
        for i in range(step_r+1):
            for j in range(step_c+1):
                score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0], j*stride[1]:j*stride[1]+kernelSize[1]]))
                I.append((i,j))
    else:
        ## there is an object, we only need to search in the neighbourhood
        for i in range(index[0]-1, index[0]+2):
            for j in range(index[1]-1, index[1]+2):
                score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0], j*stride[1]:j*stride[1]+kernelSize[1]]))
                I.append((i,j))
    # define a inertia parameter to make the window stable
    if inertia:
        iner_score = np.sum(res[inertia[0]:inertia[2], inertia[1]:inertia[3]])*(inertialParam+1)
    else:
        iner_score = 0
    # find the roi with highest score
    s = max(score)
    idx = score.index(s)
    center = np.array([[0],[0]])
    if iner_score>=s>minScore:
        roi = inertia
    elif s>minScore:
        roi = (I[idx][0]*stride[0], I[idx][1]*stride[1], I[idx][0]*stride[0]+kernelSize[0], I[idx][1]*stride[1]+kernelSize[1])   
    else:
        roi = (0,0,0,0)
        index = None
    if max(roi)!=0:
        p= np.mgrid[0:im1.shape[0], 0:im1.shape[1]]
        x = p[0,:]
        y = p[1,:]
        mask = np.zeros((im1.shape[0], im1.shape[1]))
        mask[roi[0]:roi[2], roi[1]:roi[3]] = res[roi[0]:roi[2], roi[1]:roi[3]]
        center[0] = np.sum(mask*x)/s
        center[1] = np.sum(mask*y)/s
        roi = (int(center[0]-kernelSize[0]/2), int(center[1]-kernelSize[1]/2), 
        int(center[0]+kernelSize[0]/2), int(center[1]+kernelSize[1]/2))   
        index = I[idx]
    return roi, center, index

def track(im1, im2, roi, featnumber, feat, kernelSize, minSpeed=1):
    '''
    PARAMETERS
    - roi:(1x4TUPLE) the pre-defined ROI
    - featnumber:(INT) the max number of feature points
    - feat:(None or LIST) the feature points got from the last loop(used to track)
    - kernelSize:(1x2TUPLE) the size of ROI
    - minSpeed: (FLOAT) if a pixel's displacement is larger than minSpeed, it can be seen as dynamic.(According to assumption 1)
    OUTPUTS
    - kp2:(None or LIST) feature points tracked via sparse optical flow, if there is no points, return None
    - center:(1x2TUPLE) center of ROI
    - roi:(1x4TUPLE) roi got from sparse optical flow
    '''
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    ROI_mask = np.zeros(gray1.shape)
    ROI_mask[roi[0]:roi[2], roi[1]:roi[3]] = 1
    center = np.array([[0],[0]])
    if not isinstance(feat, np.ndarray):
        orb = cv2.ORB_create(nfeatures=featnumber)
        kp1, des1 = orb.detectAndCompute(gray1, np.uint8(ROI_mask))
        feat = []
        if len(kp1): 
            for i in range(len(kp1)):
                feat.append(kp1[i].pt)
        else:
            return None, center, (int(center[0]-kernelSize[0]/2), int(center[1]-kernelSize[1]/2), 
    int(center[0]+kernelSize[0]/2), int(center[1]+kernelSize[1]/2)) 
    feat = np.array(feat, dtype=np.float32)
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat, np.uint8(ROI_mask))
    kp2 = np.array([feat2[i, :]  for i in range(feat2.shape[0]) if (status[i] and error[i]>minSpeed)], dtype=np.float32)
    if kp2.shape[0]:
        center[0] = np.mean(kp2[:,1])
        center[1] = np.mean(kp2[:,0])
    else:
        kp2 = None
    roi = (int(center[0]-kernelSize[0]/2), int(center[1]-kernelSize[1]/2), 
    int(center[0]+kernelSize[0]/2), int(center[1]+kernelSize[1]/2)) 
    return kp2, center, roi

def sptrack(im1, im2, roi, featnumber, feat, kernelSize, minSpeed=1):
    pass