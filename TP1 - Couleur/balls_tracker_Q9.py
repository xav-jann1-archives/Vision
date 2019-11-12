#!/usr/bin/env python
import cv2
import numpy as np

pctr = (-1, -1)


def track(image):
    '''Accepts BGR image as Numpy array
       Returns: (x,y) coordinates of centroid if found
                (-1,-1) if no centroid was found
                None if user hit ESC
    '''
    global pctr

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image for only pink colors
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([179, 255, 255])

    # Threshold the HSV image to get only pink colors
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Image processing
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bmask = cv2.erode(mask, element, iterations=3)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bmask = cv2.dilate(bmask, element, iterations=20)

    # Find Contours
    contours, hierarchy = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #im, contours, hierarchy = cv2.findContours(
    #    bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    showingCNTs = []  # Contours that are visible
    areas = []  # The areas of the contours

    # Find Specific contours
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if area > 300:
            areas.append(area)
            showingCNTs.append(cnt)

    # Only Highlight the largest object
    centroid_x, centroid_y = None, None

    # Get the closest centroid to previous one:
    for cnt in showingCNTs:

        # Take the moments to get the centroid
        x, y = getCentroid(cnt)

        d = 100
        if pctr[0] == -1 and pctr[1] == -1:
            centroid_x, centroid_y = x, y
        elif abs(x - pctr[0]) < d and abs(y - pctr[1]) < d:
            centroid_x, centroid_y = x, y

    # Assume no centroid
    ctr = (-1, -1)

    # Use centroid if it exists
    if centroid_x != None and centroid_y != None:
        ctr = (centroid_x, centroid_y)

        # Put black circle in at centroid in image
        cv2.circle(image, ctr, 10, (0, 0, 125), -1)

    # Save previous point:
    pctr = ctr

    # Display full-color image
    cv2.imshow('PinkBallTracker', image)

    # Force image display, setting centroid to None on ESC key input
    if cv2.waitKey(1) & 0xFF == 27:
        ctr = None

    # Return coordinates of centroid
    return ctr


# Get centroid from contour:
def getCentroid(cnt):
    moments = cv2.moments(cnt)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    return centroid_x, centroid_y


if __name__ == '__main__':
    capture = cv2.VideoCapture('ball4.mp4')

    while True:
        okay, image = capture.read()
        if okay:
            if not track(image):
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            print('Capture failed')
            break
