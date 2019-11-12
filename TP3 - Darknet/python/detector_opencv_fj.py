# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

from scipy.misc import imread
import cv2
import darknet as dn

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    boxes = dn.make_boxes(net)
    probs = dn.make_probs(net)
    num =   dn.num_boxes(net)
    dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
    return res


import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))



# Darknet
#net = dn.load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
#meta = dn.load_meta("cfg/coco.data")
#dn.set_gpu(0) to use only if gpu available
net = dn.load_net("/home/astro/darknet/cfg/yolov3.cfg", "/home/astro/darknet/cfg/yolov3.weights", 0)
meta = dn.load_meta("/home/astro/darknet/cfg/coco.data")
raw_input(' network loaded in GPU , Press Enter')
r = dn.detectfj(net, meta, "/home/astro/darknet/data/dog.jpg")
print "test direct path for image :"
print r

## scipy
arr= imread('data/dog.jpg')
im = array_to_image(arr)
dn.rgbgr_image(im)
r = dn.detectfj(net, meta, im)
print "test direct scipy read for image :"
print r

# OpenCV
arr = cv2.imread('/home/astro/darknet/data/dog.jpg')
im = array_to_image(arr)
dn.rgbgr_image(im)
r = dn.detectfj(net, meta, im)
print "test direct opencv (cv2) read for image :"
print r

