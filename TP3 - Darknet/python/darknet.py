from ctypes import *
import math
import random
import time
import cv2
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/fs03/share/users/xavier.jannin/home/Bureau/Vision/TP3_Darknet/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detectfj(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    
    if isinstance(image, bytes):  
        # image is a filename 
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:  
        # image is a numpy array 
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        #rm fj test im = array_to_image(image)
        #rm fj test rgbgr_image(im)
	im = array_to_image(image)

    im_modified = image
    
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, 
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    
    for j in range(num):
        for i in range(meta.classes):
	    if dets[j].prob[i] > 0:
	      print meta.names[i], dets[j].prob[i], meta.names[i] in colors_dict.keys()
	    #print (meta.names[i] in colors_dict.values())
            if dets[j].prob[i] > 0 and meta.names[i] in colors_dict.keys():
                b = dets[j].bbox
                x, y, w, h = b.x, b.y, b.w, b.h
                
                res.append((meta.names[i], dets[j].prob[i], (x, y, w, h)))
                  
                color = colors_dict[meta.names[i]]
                cv2.rectangle(im_modified, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
                
                cv2.putText(im_modified, meta.names[i], (int(x - w/2), int(y - h/2 - 10)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 2)
                
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)    
    
    return res, im_modified



def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    
    im_modified = cv2.imread(image)
    
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []

    for j in range(num):
        for i in range(meta.classes):
	    if dets[j].prob[i] > 0:
	      print meta.names[i], dets[j].prob[i], meta.names[i] in colors_dict.keys()
	    #print (meta.names[i] in colors_dict.values())
            if dets[j].prob[i] > 0 and meta.names[i] in colors_dict.keys():
                b = dets[j].bbox
                x, y, w, h = b.x, b.y, b.w, b.h
                
                res.append((meta.names[i], dets[j].prob[i], (x, y, w, h)))
                  
                color = colors_dict[meta.names[i]]
                cv2.rectangle(im_modified, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
                
                cv2.putText(im_modified, meta.names[i], (int(x - w/2), int(y - h/2 - 10)), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 2)
    
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    
    cv2.imshow("Image",im_modified)
    #cv2.waitKey(0)
    
    return res

def load_model():
    net = load_net("cfg/yolov3-tiny.cfg", "yolov3-tiny.weights", 0)
    meta = load_meta("cfg/coco.data")
    return net,meta

colors_dict = {
  "dog": (83,128,168),
  "person": (239,117,239),
  "car": (50, 50, 50),
  "bird": (10, 240, 10),
  "truck": (0,0,0),
  "horse": (240,10,10),
  "chair": (20,20,240)
}

video_capture = cv2.VideoCapture(0)
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    
    net,meta = load_model()
    # Image detection:
    #r = detect(net, meta, "data/person.jpg")
    
    # Live detection:
    while True:
    
      # Capture frame-by-frame:
      ret, frame = video_capture.read()
      
      #cv2.imshow("Image", frame)

      
      r, img = detectfj(net, meta, frame)
      print r
      
      cv2.imshow("Image",img)
    
      
      # Display the resulting frame
      #cv2.imshow('Video', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
	break
