#!/usr/bin/env python
import cv2
import numpy as np

# For OpenCV2 image display
WINDOW_NAME = 'ColorBallTracker' 

lower_color = []
upper_color = []

def track(image):

    '''Accepts BGR image as Numpy array
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5,5),0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow(WINDOW_NAME, mask)
    
    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5,5),0)

    # Take the moments to get the centroid
    moments = cv2.moments(bmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    # Assume no centroid
    ctr = (-1,-1)

    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:
        ctr = (centroid_x, centroid_y)

        # Put black circle in at centroid in image
        cv2.circle(image, ctr, 10, (0,0,125), -1)

    # Display full-color image
    cv2.imshow(WINDOW_NAME, image)
    #cv2.imshow(WINDOW_NAME, mask)

    # Force image display, setting centroid to None on ESC key input
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None
    
    # Return coordinates of centroid
    return ctr



# Test with input from camera
if __name__ == '__main__':

    capture = cv2.VideoCapture('ball3.mp4')
    
    okay, image = capture.read()
    blur = cv2.GaussianBlur(image, (5,5),0)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.imshow(WINDOW_NAME, blur)
    h, w, n = np.shape(hsv)
    
    #c = hsv[h//2][w//2]
    c = hsv[384][850]
    
    print(c)
    
    lower_color = np.array([c[0] - 10, c[1] - 50, c[2] - 50])
    upper_color = np.array([c[0] + 10, c[1] + 50, c[2] + 50])
    #lower_color = np.array([c[0] - 10, 100, 100])
    #upper_color = np.array([c[0] + 10, 255, 255])

    ...

    while True:
        okay, image = capture.read()

        if okay:
            if not track(image): break
            if cv2.waitKey(1) & 0xFF == 27: break

        else:
           print('Capture failed')
           break


