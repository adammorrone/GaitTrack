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
orangeLower = (0, 89, 211)
orangeUpper = (23, 255, 255)
greenLower = (52, 39, 112)
greenUpper = (89, 255, 255)
blueLower = (88, 128, 94)
blueUpper = (117, 255, 255)
orange_pts = deque()
blue_pts = deque()
green_pts = deque()
blue_data = np.array([[], []])
orange_data = np.array([[], []])
green_data = np.array([[], []])

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
    frame = imutils.resize(frame, width=1200)

    blurblue = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurblue, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    orange_mask = cv2.inRange(hsv, orangeLower, orangeUpper)
    blue_mask = cv2.inRange(hsv, blueLower, blueUpper)
    green_mask = cv2.inRange(hsv, greenLower, greenUpper)

    orange_mask = cv2.erode(orange_mask, None, iterations=2)
    orange_mask = cv2.dilate(orange_mask, None, iterations=2)

    blue_mask = cv2.erode(blue_mask, None, iterations=2)
    blue_mask = cv2.dilate(blue_mask, None, iterations=2)

    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    orange_cnts = cv2.findContours(orange_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    orange_cnts = imutils.grab_contours(orange_cnts)
    orange_center = None

    blue_cnts = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    blue_cnts = imutils.grab_contours(blue_cnts)
    blue_center = None
    
    green_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    green_cnts = imutils.grab_contours(green_cnts)
    green_center = None

    # only proceed if at least one contour was found
    if len(orange_cnts) > 0 and len(blue_cnts) > 0 and len(green_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(orange_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        orange_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(frame, orange_center, 5, (0, 0, 255), -1)

        orange_data = [np.append(orange_data[0], [int(M["m10"] / M["m00"])]),
                    np.append(orange_data[1], [int(M["m01"] / M["m00"])])]


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

        blue_data = [np.append(blue_data[0], [int(M["m10"] / M["m00"])]),
                     np.append(blue_data[1], [int(M["m01"] / M["m00"])])]

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(green_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        green_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.circle(frame, green_center, 5, (0, 0, 255), -1)

        green_data = [np.append(green_data[0], [int(M["m10"] / M["m00"])]),
                     np.append(green_data[1], [int(M["m01"] / M["m00"])])]



    # update the points queue
    orange_pts.appendleft(orange_center)
    blue_pts.appendleft(blue_center)
    green_pts.appendleft(green_center)



    # loop over the set of tracked points
    for i in range(1, len(orange_pts)):
        # if either of the tracked points are None, ignore
        # them
        if orange_pts[i - 1] is None or orange_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 3)
        cv2.line(frame, orange_pts[i - 1], orange_pts[i], (4, 236, 255), thickness)

    for i in range(1, len(blue_pts)):
        # if either of the tracked points are None, ignore
        # them
        if blue_pts[i - 1] is None or blue_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 3)
        cv2.line(frame, blue_pts[i - 1], blue_pts[i], (255, 150, 50), thickness)

    for i in range(1, len(green_pts)):
        # if either of the tracked points are None, ignore
        # them
        if green_pts[i - 1] is None or green_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 3)
        cv2.line(frame, green_pts[i - 1], green_pts[i], (255, 150, 50), thickness)

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

fig, ax = plt.subplots(2, 3)

t_blue_x = np.arange(0, len(blue_data[0][12:]), 1)
t_orange_x = np.arange(0, len(orange_data[0][12:]), 1)
t_green_x = np.arange(0, len(green_data[0][12:]), 1)

t_blue_y = np.arange(0, len(blue_data[1][12:]), 1)
t_orange_y = np.arange(0, len(orange_data[1][12:]), 1)
t_green_y = np.arange(0, len(green_data[1][12:]), 1)


ax[0][0].set(xlabel='frame', ylabel='x position',
       title='blue Marker X Position')

ax[1, 0].set(xlabel='frame', ylabel='x position',
       title='orange Marker X Position')


ax[0][1].set(xlabel='frame', ylabel='y position',
       title='blue Marker Y Position')

ax[1, 1].set(xlabel='frame', ylabel='y position',
       title='orange Marker Y Position')

ax[0, 0].grid()
ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()

ax[0, 0].plot(t_blue_x, blue_data[0][12:])
ax[1, 0].plot(t_orange_x, orange_data[0][12:])
ax[0, 1].plot(t_blue_y, blue_data[1][12:])
ax[1, 1].plot(t_orange_y, orange_data[1][12:])

fig_delta, ax_delta = plt.subplots(2, 2)

ax_delta[0, 0].grid()
ax_delta[0, 1].grid()
ax_delta[1, 0].grid()
ax_delta[1, 1].grid()

ax_delta[0, 1].plot(t_blue_x, blue_data[0][12:]-orange_data[0][12:])
ax_delta[1, 1].plot(t_orange_x, blue_data[1][12:]-orange_data[1][12:])

ax_delta[0, 0].plot(t_green_x, blue_data[0][12:]-green_data[0][12:])
ax_delta[1, 0].plot(t_green_x, blue_data[1][12:]-green_data[1][12:])

ax_delta[0, 1].set(xlabel='frame', ylabel='delta x',
       title='Delta X Position Blue - Orange')

ax_delta[1, 1].set(xlabel='frame', ylabel='delta y',
       title='Delta Y Position Blue - Orange')

ax_delta[0, 0].set(xlabel='frame', ylabel='delta x',
       title='Delta X Position Blue - Green')

ax_delta[1, 0].set(xlabel='frame', ylabel='delta y',
       title='Delta Y Position Blue - Green')

plt.show()

