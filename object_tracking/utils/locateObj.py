#!/usr/bin/python

#np.ceil(x) is used to get the minimum integer that equal or larger than x
#np.floor(x) is used to get the minimum interger that equal or smaller than x
#flow[...,0] means the value of index 0 of last dimension 

import cv2
import numpy as np

'''
# this function is to evaluate the score of ROI
def evaluate(flow,minSpeed,horiParam,Euclian,eroRate,dilRate):
    # ??? 
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else (horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1]))/2
    # construct a matrix and its values are all 1, which can be used to calculate the score!
    res = np.uint8(np.ones_like(flow))
    # if flow is lower than minSpeed, assign 0
    res[flow<minSpeed] = 0
    # open operation firstly do erosion then do dilation, which can decreasse small noise and make the contour sharp
    # np.ones((eroRate,eroRate)) is used to construct a kernel 
    res = cv2.erode(res,np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res,np.ones((dilRate,dilRate),np.uint8))
    return np.sum(res)

# this function is to acquire the ROI and the heighest score of current ROI
def conv(img1,img2,kernelSize,stride,inertia,minScore=100,eroRate=25,dilRate=25,Euclian=False,horiParam=0.8,minSpeed=1,inertiaParam=2):
    #exception handling
    assert len(img1.shape) == 2, \
    f'Gray images required'
    #to get the size of image 
    r, c = img1.shape[:2]
    # determine how many steps
    step_r = int(np.floor((r-kernelSize[0])/stride[0]))
    step_c = int(np.floor((c-kernelSize[1])/stride[1]))
    score, x, y = [], [], []
    for i in range(step_c + 1):
        for j in range(step_r + 1):
            #construct region of interest and it should be acquired by iterating all the images
            roi1 = img1[i*stride[0]:i*stride[0]+kernelSize[0],j*stride[1]:j*stride[1]+kernelSize[1]]
            roi2 = img2[i*stride[0]:i*stride[0]+kernelSize[0],j*stride[1]:j*stride[1]+kernelSize[1]]
            #parameter dictionary
            opParam = dict(pyr_scale=0.5,levels=5,winsize=15,iterations=5,poly_n=1,poly_sigma=0,flags=0)
            #calculate the optical flow using openCV build in method
            flow = cv2.calcOpticalFlowFarneback(roi1,roi2,None,**opParam)
            #evaluate current quality and return the score and add it to the list
            score.append(evaluate(flow,minSpeed,horiParam,Euclian,eroRate,dilRate))
            #all the x and y coordinates
            x.append(i*stride[0])
            y.append(j*stride[1])
    #define an inertia parameter to make the window stable
    if inertia:
        flow = cv2.calcOpticalFlowFarneback(img1[inertia[0]:inertia[2],inertia[1]:inertia[3]],img2[inertia[0]:inertia[2],inertia[1]:inertia[3]],None,**opParam)
        iner_score = evaluate(flow,minSpeed,horiParam,Euclian,eroRate,dilRate)*inertiaParam
    else:
        iner_score = 0
    # find the roi with highest score
    s = max(score)
    idx = score.index(s)
    if iner_score >= s > minScore:
        roi = inertia
    elif s > minScore:
        roi = (x[idx],y[idx],x[idx]+kernelSize[0],y[idx]+kernelSize[1])
    else:
        roi = (0,0,0,0)
    return score, roi 
'''

'''
def conv(im1, im2, kernelSize, stride, inertia=None, minScore=100, eroRate=15, dilRate=25, winsize=15, Euclian=False, horiParam=0.8, minSpeed=1, inertialParam=1.5):
    r, c = im1.shape[:2]
    gray1, gray2 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
    # create flow map
    opParam = dict(pyr_scale=0.5, levels=5, winsize=winsize, iterations=5, poly_n=1, poly_sigma=0, flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **opParam)
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else (horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1]))/2
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed]= 0
    res = cv2.erode(res, np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res, np.ones((dilRate,dilRate),np.uint8))
    # generate the kernel 
    step_r = int(np.ceil((r-kernelSize[0])/stride[0]))
    step_c = int(np.ceil((c-kernelSize[1])/stride[1]))
    score, x, y = [], [], []
    for i in range(step_r+1):
        for j in range(step_c+1):
            score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0], j*stride[1]:j*stride[1]+kernelSize[1]]))
            x.append(i*stride[0])
            y.append(j*stride[1])
    # define a inertia parameter to make the window stable
    if inertia:
        iner_score = np.sum(res[inertia[0]:inertia[2], inertia[1]:inertia[3]])*inertialParam
    else:
        iner_score = 0
    # find the roi with highest score
    s = max(score)
    idx = score.index(s)
    if iner_score>=s>minScore:
        roi = inertia
    elif s>minScore:
        roi = (x[idx], y[idx], x[idx]+kernelSize[0], y[idx]+kernelSize[1])       
    else:
        roi = (0,0,0,0)
    return score, roi
'''

def conv(im1, im2, kernelSize, stride, inertia=None, minScore=100, eroRate=21, dilRate=25, winsize=15, Euclian=False, horiParam=0.8, minSpeed=1, inertialParam=1.5):
    r, c = im1.shape[:2]
    gray1, gray2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # create flow map
    opParam = dict(pyr_scale=0.5, levels=5, winsize=winsize, iterations=5, poly_n=1, poly_sigma=0, flags=0)
    # use dense optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **opParam)
    # flow returned by dense optical flow function contains distance values both in x and y direction 
    # what we want here is to acquire scale value in x and y direction separately
    # if we use Euclidian distance, it is the sqrt sum of square value in x and y direction
    # else it is weighted average value of x and y direction's data
    # Due to the larger movement in x direction of objects in this video, we should set weight of x larger 
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1])
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed]= 0
    res = cv2.erode(res, np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res, np.ones((dilRate,dilRate),np.uint8))
    # generate the kernel 
    step_r = int(np.ceil((r-kernelSize[0])/stride[0]))
    step_c = int(np.ceil((c-kernelSize[1])/stride[1]))
    score, x, y = [], [], []
    # for i in range(step_r) also can work here
    for i in range(step_r+1):
        for j in range(step_c+1):
            score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0], j*stride[1]:j*stride[1]+kernelSize[1]]))
            x.append(i*stride[0])
            y.append(j*stride[1])
    # define an inertia parameter to make the window stable
    # inertia has two options: None or ROI and None means we don't open inertia mode
    # otherwise we open inertia mode and the following ROI can remain the previous frame's ROI so that 
    # the window moves with a very slow speed and inertiaParam is used to adjust the level.
    # If inertiaParam is too large, the window stop moving nearly so it shouldn't be too large!
    if inertia:
        iner_score = np.sum(res[inertia[0]:inertia[2], inertia[1]:inertia[3]])*inertialParam
    else:
        iner_score = 0
    # find the roi with highest score
    s = max(score)
    idx = score.index(s)
    center = [0,0]
    if iner_score>=s>minScore:
        roi = inertia
    elif s>minScore:
        roi = (x[idx], y[idx], x[idx]+kernelSize[0], y[idx]+kernelSize[1])
        p= np.mgrid[0:im1.shape[0], 0:im1.shape[1]]
        x = p[0,:]
        y = p[1,:]
        mask = np.zeros((im1.shape[0], im1.shape[1]))
        mask[roi[0]:roi[2], roi[1]:roi[3]] = res[roi[0]:roi[2], roi[1]:roi[3]]
        center[0] = np.sum(mask*x)/s
        center[1] = np.sum(mask*y)/s
        roi = (int(center[0]-kernelSize[0]/2), int(center[1]-kernelSize[1]/2), 
        int(center[0]+kernelSize[0]/2), int(center[1]+kernelSize[1]/2))      
    else:
        roi = (0,0,0,0)
    return score, roi, center
