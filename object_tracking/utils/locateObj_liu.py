import cv2
import numpy as np

def conv(img1,img2,kernelSize,stride,inertia=None,minScore=100,eroRate=25,dilRate=25,winSize=15,Euclian=False,horiParam=0.8,minSpeed=1,inertialParam=1.5):
    r, c = img1.shape[:2]
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    opParam = dict(pyr_scale=0.5,levels=5,winsize=winSize,iterations=5,poly_n=5,poly_sigma=1.1,flags=0)
    flow = cv2.calcOpticalFlowFarneback(gray1,gray2,None,**opParam)
    flow = np.sqrt((flow[...,0]**2+flow[...,1]**2)) if Euclian else horiParam*abs(flow[...,0])+(1-horiParam)*abs(flow[...,1])
    res = np.uint8(np.ones_like(flow))
    res[flow<minSpeed] = 0
    res = cv2.erode(res,np.ones((eroRate,eroRate),np.uint8))
    res = cv2.dilate(res,np.ones((dilRate,dilRate),np.uint8))
    #generate a kernel
    step_r = int(np.ceil((r-kernelSize[0])/stride[0]))
    step_c = int(np.ceil((r-kernelSize[1])/stride[1]))
    score,x,y = [],[],[]
    for i in range(step_r+1):
        for j in range(step_c+1):
            score.append(np.sum(res[i*stride[0]:i*stride[0]+kernelSize[0],j*stride[1]:j*stride[1]+kernelSize[1]]))
            x.append(i*stride[0])
            y.append(j*stride[1])
    # define an inertia parameter to make the window stable
    if inertia:
        iner_score = np.sum(res[inertia[0]:inertia[2], inertia[1]:inertia[3]])*inertialParam
    else:
        iner_score = 0
    #find the highest score
    s = max(score)
    idx = score.index(s)
    center = [0,0]
    if iner_score>=s>minScore:
        roi = inertia
    elif s > minScore:
        roi = (x[idx],y[idx],x[idx]+kernelSize[0],y[idx]+kernelSize[1])
        p = np.mgrid[0:img1.shape[0],0:img1.shape[1]]
        x = p[0,:]
        y = p[1,:]
        mask = np.zeros((img1.shape[0],img1.shape[1]))
        mask[roi[0]:roi[2],roi[1]:roi[3]] = res[roi[0]:roi[2],roi[1]:roi[3]]
        center[0] = np.sum(mask*x)/s
        center[1] = np.sum(mask*y)/s
        roi = (int(center[0]-kernelSize[0]/2),int(center[1]-kernelSize[1]/2)),
        int(center[0]+kernelSize[0]/2),int(center[1]+kernelSize[1]/2)
    else:
        roi = (0,0,0,0)
    return score, roi, center
        


    