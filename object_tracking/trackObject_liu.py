from utils_liu import trackObj
import cv2

# Dense Tracker
'''
Green box means "I find a moving object but I am not sure if it is ture."
Red box means "I am sure that this is a moving object."
Yellow box means "I found the target and now it is under occlusion, so this is the predicted position based on the history"
'''

path = '../RectifiedImgOcc/right'
size = (1024,729)
cvt = cv2.COLOR_BGR2RGB
# the target path under occulusion is predicted based on the previous 20 frames
monsize = 20
thresh = [10,3]
# window size
kernel = (200,200)
# step size
stride = (20,20)

def dense():
    # define 2 modes sampling=True means fps is high but the video will lose some frames
    sampling = [True,False]
    # create an object of DenseTracker  
    t = trackObj.DenseTracker(path,size,cvt,monsize,thresh,kernel,stride,sampling=sampling[0],rec=False)
    # call the member function loop to handle pictures
    t.loop(100,0,False)

def sparse():
    path = '../RectifiedImgOcc/left'
    featNum = 10000
    sampling = False
    t = trackObj.SparseTracker(path,size,cvt,monsize,thresh,kernel,stride,sampling=sampling[1],rec=False)
    t.loop(100,0)


if __name__ == '__main__':
    dense()
    #sparse()








