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
greenLower = (54, 36, 68)
greenUpper = (90, 180, 161)

redLower = (118, 80, 96)
redUpper = (205, 201, 234)
green_pts = deque()
green_pts2 = deque()
red_pts = deque()
red_data = np.array([[], []])
green_data = np.array([[], []])
green_data2 = np.array([[], []])

NUM_DOTS = 2

# if a video path was not supplied, grab the reference
# to the webcam
#if not args.get("video", False):
#vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
#else:
#vs = cv2.VideoCapture(args["video"])

vs = cv2.VideoCapture(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\robo_footage\two_green.mp4')



# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] #if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1200)

    cv2.imshow("Frame", frame)


    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    cv2.imshow("Frame", frame)


    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    green_mask = cv2.inRange(hsv, greenLower, greenUpper)
    red_mask = cv2.inRange(hsv, redLower, redUpper)

    cv2.imshow("Frame", green_mask)

    green_mask = cv2.erode(green_mask, None, iterations=2)

    cv2.imshow("Frame", green_mask)

    green_mask = cv2.dilate(green_mask, None, iterations=5)

    cv2.imshow("Frame", green_mask)

    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)


    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    green_cnts = cv2.findContours(green_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

#    cv2.imshow("Frame", green_cnts)

    green_cnts = imutils.grab_contours(green_cnts)

    green_center = None

    # only proceed if at least one contour was found
    if len(green_cnts) > 0:
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

    green_data = [np.append(green_data[0], [int(M["m10"] / M["m00"])]),
                  np.append(green_data[1], [int(M["m01"] / M["m00"])])]


    if len(green_cnts) > 1:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = min(green_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        green_center2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)

    green_data2 = [np.append(green_data2[0], [int(M["m10"] / M["m00"])]),
                  np.append(green_data2[1], [int(M["m01"] / M["m00"])])]

    red_cnts = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    red_cnts = imutils.grab_contours(red_cnts)
    red_center = None

    # only proceed if at least one contour was found
    if len(red_cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(red_cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        red_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 8:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)

    red_data = [np.append(red_data[0], [int(M["m10"] / M["m00"])]), np.append(red_data[1], [int(M["m01"] / M["m00"])])]

    # update the points queue
    green_pts.appendleft(green_center)
    green_pts2.appendleft(green_center2)
    red_pts.appendleft(red_center)

    thickness = 1

    # loop over the set of tracked points
    for i in range(1, len(green_pts)):
        # if either of the tracked points are None, ignore
        # them
        if green_pts[i - 1] is None or green_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        cv2.line(frame, green_pts[i - 1], green_pts[i], (4, 236, 255), thickness)

    for i in range(1, len(green_pts2)):
        # if either of the tracked points are None, ignore
        # them
        if green_pts2[i - 1] is None or green_pts2[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        cv2.line(frame, green_pts2[i - 1], green_pts2[i], (4, 236, 255), thickness)

    for i in range(1, len(red_pts)):
        # if either of the tracked points are None, ignore
        # them
        if red_pts[i - 1] is None or red_pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        cv2.line(frame, red_pts[i - 1], red_pts[i], (0, 0, 255), thickness)



    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
# if not args.get("video", False):
#     vs.stop()
#
# # otherwise, release the camera
# else:
vs.release()

# close all windows
#cv2.destroyAllWindows()

fig, ax = plt.subplots(2, 2)
t_red_x = np.arange(0, len(red_data[0][12:]), 1)
t_green_x = np.arange(0, len(green_data[0][12:]), 1)
t_red_y = np.arange(0, len(red_data[1][12:]), 1)
t_green_y = np.arange(0, len(green_data[1][12:]), 1)


ax[0][0].set(xlabel='frame', ylabel='x position',
       title='Red Marker X Position')

ax[1, 0].set(xlabel='frame', ylabel='x position',
       title='green Marker X Position')


ax[0][1].set(xlabel='frame', ylabel='y position',
       title='Red Marker Y Position')

ax[1, 1].set(xlabel='frame', ylabel='y position',
       title='green Marker Y Position')

ax[0, 0].grid()
ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()

ax[0, 0].plot(t_red_x, red_data[0][12:])
ax[1, 0].plot(t_green_x, green_data[0][12:])
ax[0, 1].plot(t_red_y, red_data[1][12:])
ax[1, 1].plot(t_green_y, green_data[1][12:])



fig_delta, ax_delta = plt.subplots(1, 2)

ax_delta[0].grid()
ax_delta[1].grid()

ax_delta[0].plot(t_red_x, red_data[0][12:]-green_data[0][12:])
ax_delta[1].plot(t_green_x, red_data[1][12:]-green_data[1][12:])

ax_delta[0].set(xlabel='frame', ylabel='delta x',
       title='Delta X Position Between Markers')

ax_delta[1].set(xlabel='frame', ylabel='delta y',
       title='Delta Y Position Between Markers')

plt.show()

