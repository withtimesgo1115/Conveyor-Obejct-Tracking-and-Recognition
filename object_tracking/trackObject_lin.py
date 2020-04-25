from utils import trackObj
import cv2

# Dense Tracker
'''
GREEN box means "I find a moving object, but I'm not sure it is the target"
RED box means "I'm sure that this is the target"
YELLOW box means "I found the target and now it is under occlusion, so this is the position I guess based on the history"
'''
path= '../RectifiedImgOcc/right/'
size = (1024, 729) # unnecessary
cvt = cv2.COLOR_BGR2RGB
monsize = 20 # the target path under occulusion is predicted based on the previous 20 frames
thresh = [10, 3]
kernel = (250, 250)
stride = (30, 30)
def dense():
    sampling = True # if True the fps is much higher, but the video will lose some frames
    t = trackObj.DenseTracker(path, size, cvt, monsize, thresh, kernel, stride, sampling=True, rec=False)
    t.loop(100, 0, False)

# Sparse Tracker
def sparse():
    path= '../RectifiedImgOcc/left/'
    # without sampling
    kernel = (200, 200)
    stride = (100, 100)
    featNum = 10000
    sampling = False
    t = trackObj.SparseTracker(path, size, cvt, monsize, thresh, kernel, stride, featNum, sampling=False, rec=False)
    t.loop(100, 0)

if __name__=='__main__':
    dense()
    #sparse()