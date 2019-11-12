#!/usr/bin/env python

'''
Track a green ball using OpenCV.

    Copyright (C) 2015 Conan Zhao and Simon D. Levy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as 
    published by the Free Software Foundation, either version 3 of the 
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License 
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import cv2
import numpy as np

# For OpenCV2 image display
WINDOW_NAME = 'BallTracker' 


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

    # Threshold the HSV image for only one color
    [lower_color, upper_color] = getThreshold('blue')
    
    # Threshold the HSV image to get only one color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
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


# Get threshold for each color (blue, green, pink or yellow):
def getThreshold(color):
    
    if color == 'blue':
        # Returns directly threshold:
        lower_blue = np.array([80, 140, 140])
        upper_blue = np.array([120, 255, 255])
        return [lower_blue, upper_blue]

    elif color == 'green': bgr_color = [135, 161, 99]
    elif color == 'pink': bgr_color = [86, 70, 167]
    elif color == 'yellow': bgr_color = [160, 219, 200]
    else:
      print("unknown color for threshold: default 'green'")
      bgr_color = [135, 161, 99]
    
    # Get HSV value from BGR color:
    bgr_color = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)
    
    # Get Hue value:
    h_color = hsv_color[0][0][0]
    
    # Lower bound:
    if h_color < 20:
      h_color = 20
    
    # Threshold:
    lower_color = np.array([h_color - 20, 80, 80])
    upper_color = np.array([h_color + 20, 255, 255])
    
    return [lower_color, upper_color]


# Test with input from camera
if __name__ == '__main__':

    capture = cv2.VideoCapture('ball3.mp4')

    while True:
        okay, image = capture.read()
        if okay:
            if not track(image): break
            if cv2.waitKey(1) & 0xFF == 27: break
        else:
           print('Capture failed')
           break
