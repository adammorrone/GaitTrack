# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# define the  lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# ball in the HSV color space, then initialize the
# list of tracked points
#yellowLower = (28, 48, 85)
#yellowUpper = (62, 182, 255)

yellowLower = (24, 53, 69)
yellowUpper = (60, 255, 255)
blueLower = (88, 128, 94)
blueUpper = (117, 255, 255)
yellow_pts = deque()
blue_pts = deque()
blue_data = np.array([[], []])
yellow_data = np.array([[], []])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1400)

    blurblue = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurblue, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    yellow_mask = cv2.inRange(hsv, yellowLower, yellowUpper)
    blue_mask = cv2.inRange(hsv, blueLower, blueUpper)

    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)

    blue_mask = cv2.erode(blue_mask, None, iterations=2)
    blue_mask = cv2.dilate(blue_mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    yellow_cnts = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    yellow_cnts = imutils.grab_contours(yellow_cnts)
    yellow_center = None

    # only proceed if at least one contour was found
    if len(yellow_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(yellow_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        yellow_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(frame, yellow_center, 5, (0, 0, 255), -1)

    blue_cnts = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blue_cnts = imutils.grab_contours(blue_cnts)
    blue_center = None

    yellow_data = [np.append(yellow_data[0], [int(M["m10"] / M["m00"])]), np.append(yellow_data[1], [int(M["m01"] / M["m00"])])]

    # only proceed if at least one contour was found
    if len(blue_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(blue_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        blue_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(frame, blue_center, 5, (0, 0, 255), -1)


    # update the points queue
    yellow_pts.appendleft(yellow_center)
    blue_pts.appendleft(blue_center)
    blue_data = [np.append(blue_data[0], [int(M["m10"] / M["m00"])]), np.append(blue_data[1], [int(M["m01"] / M["m00"])])]


    # loop over the set of tracked points
    for i in range(1, len(yellow_pts)):
        # if either of the tracked points are None, ignore
        # them
        if yellow_pts[i - 1] is None or yellow_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 3)
        cv2.line(frame, yellow_pts[i - 1], yellow_pts[i], (4, 236, 255), thickness)

    for i in range(1, len(blue_pts)):
        # if either of the tracked points are None, ignore
        # them
        if blue_pts[i - 1] is None or blue_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 3)
        cv2.line(frame, blue_pts[i - 1], blue_pts[i], (255, 150, 50), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
#cv2.destroyAllWindows()

fig, ax = plt.subplots(2, 2)
t_blue_x = np.arange(0, len(blue_data[0][12:]), 1)
t_yellow_x = np.arange(0, len(yellow_data[0][12:]), 1)
t_blue_y = np.arange(0, len(blue_data[1][12:]), 1)
t_yellow_y = np.arange(0, len(yellow_data[1][12:]), 1)


ax[0][0].set(xlabel='frame', ylabel='x position',
       title='blue Marker X Position')

ax[1, 0].set(xlabel='frame', ylabel='x position',
       title='Yellow Marker X Position')


ax[0][1].set(xlabel='frame', ylabel='y position',
       title='blue Marker Y Position')

ax[1, 1].set(xlabel='frame', ylabel='y position',
       title='Yellow Marker Y Position')

ax[0, 0].grid()
ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()

ax[0, 0].plot(t_blue_x, blue_data[0][12:])
ax[1, 0].plot(t_yellow_x, yellow_data[0][12:])
ax[0, 1].plot(t_blue_y, blue_data[1][12:])
ax[1, 1].plot(t_yellow_y, yellow_data[1][12:])



fig_delta, ax_delta = plt.subplots(1, 2)

ax_delta[0].grid()
ax_delta[1].grid()

ax_delta[0].plot(t_blue_x, blue_data[0][12:]-yellow_data[0][12:])
ax_delta[1].plot(t_yellow_x, blue_data[1][12:]-yellow_data[1][12:])

ax_delta[0].set(xlabel='frame', ylabel='delta x',
       title='Delta X Position Between Markers')

ax_delta[1].set(xlabel='frame', ylabel='delta y',
       title='Delta Y Position Between Markers')

plt.show()

