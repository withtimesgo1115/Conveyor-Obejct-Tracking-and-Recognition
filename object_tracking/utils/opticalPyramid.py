import cv2
import numpy as np

def layer(img1,img2,winSize=15,eroRate=15,dilRate=25,minSpeed=1,horiParam=0.8,Euclian=False):
    opParam = dict(pyr_scale=0.5,levels=5,winsize=winSize,iterations=5,poly_n=1,poly_sigma=0,flags=0)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1,gray2,None,**opParam)
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else (horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1]))/2
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed]=0
    res = cv2.erode(res,np.ones((eroRate,eroRate),np.uint8))
    score = np.sum(res)
    grid = np.mgrid[0:img1.shape[0],0:img1.shape[1]]
    return res, score

def opticalPyramid(img1,img2,level=1,iniSize=(1024,729),spliter=(1,1),minScore=100,cruel=True):
    assert level<6, \
    f'level error'
    size = iniSize
    res, score = layer(img1,img2)
    if cruel:
        #remove high speed object
        res_h, _ = layer(img1,img2,eroRate=0,minSpeed=1.5)
        res = res - res_h
        score = np.sum(res)
        grid = np.mgrid[0:img1.shape[0],0:img1.shape[1]]
    if score > minScore:
        x,y,w,h = cv2.boundingRect(res)
    return (x,y,w,h)