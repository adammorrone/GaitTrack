# import the necessary packages

import cv2
import imutils
import matplotlib.pyplot as plt
from dataclasses import dataclass

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video",
#                 help="path to the (optional) video file")
# ap.add_argument("-b", "--buffer", type=int, default=64,
#                 help="max buffer size")
# args = vars(ap.parse_args())

# define the  lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# ball in the HSV color space, then initialize the
# list of tracked points


@dataclass
class Color:
    label: str
    lower_bound_HSV: tuple
    upper_bound_HSV: tuple
    num_objects: int = 1
    x_pos: list = None
    y_pos: list = None


# list of Color objects to look for
setup = [Color("G", (54, 36, 68), (90, 180, 161), 2),
         Color("R", (118, 80, 96), (205, 201, 234), 1),
         Color("Y", (24, 65, 110), (44, 137, 255), 1),
         Color("B", (110, 20, 0), (255, 255, 131), 1)]

total_objects = 0
for color in setup:
    color.x_pos = [[None]] * color.num_objects
    color.y_pos = [[None]] * color.num_objects
    total_objects = total_objects + color.num_objects


vs = cv2.VideoCapture(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\robo_footage\two_green.mp4')

# loop through frames
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1]

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1200)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask

    for color in setup:
        mask = cv2.inRange(hsv, color.lower_bound_HSV, color.upper_bound_HSV)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=6)
        mask = cv2.erode(mask, None, iterations=4)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        center = None

        # only proceed if all objects are accounted for
        if len(cnts) == color.num_objects:
            # sort contours by y position, then use
            # them to compute the minimum enclosing circle and
            # centroid

            cnts.sort(key=lambda y_pos: cv2.moments(y_pos)['m01']/cv2.moments(y_pos)['m00'])

            for i in range(len(cnts)):
                ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
                M = cv2.moments(cnts[i])
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (255, 0, 0), 2)
                cv2.putText(frame, color.label + str(i + 1), (int(x), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                color.x_pos[i].append(int(M["m10"] / M["m00"]))
                color.y_pos[i].append(int(M["m01"] / M["m00"]))
                print(color.x_pos[0])
                print('\n')

                # loop over the set of tracked points

    for color in setup:
        for obj in range(color.num_objects):
            for i in range(len(color.x_pos[obj])):
                # if either of the tracked points are None, ignore them

                if color.x_pos[obj][i - 1] is None or color.x_pos[obj][i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                cv2.line(frame, (color.x_pos[obj][i - 1], color.y_pos[obj][i - 1]),
                         (color.x_pos[obj][i], color.y_pos[obj][i]), (4, 236, 255), 1)

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



#
# fig, ax = plt.subplots(2, 2)
# t_red_x = np.arange(0, len(red_data[0][12:]), 1)
# t_x = np.arange(0, len(data[0][12:]), 1)
# t_red_y = np.arange(0, len(red_data[1][12:]), 1)
# t_y = np.arange(0, len(data[1][12:]), 1)
#
#
# ax[0][0].set(xlabel='frame', ylabel='x position',
#        title='Red Marker X Position')
#
# ax[1, 0].set(xlabel='frame', ylabel='x position',
#        title='green Marker X Position')
#
#
# ax[0][1].set(xlabel='frame', ylabel='y position',
#        title='Red Marker Y Position')
#
# ax[1, 1].set(xlabel='frame', ylabel='y position',
#        title='green Marker Y Position')
#
# ax[0, 0].grid()
# ax[0, 1].grid()
# ax[1, 0].grid()
# ax[1, 1].grid()
#
# ax[0, 0].plot(t_red_x, red_data[0][12:])
# ax[1, 0].plot(t_x, data[0][12:])
# ax[0, 1].plot(t_red_y, red_data[1][12:])
# ax[1, 1].plot(t_y, data[1][12:])
#
#
#
# fig_delta, ax_delta = plt.subplots(1, 2)
#
# ax_delta[0].grid()
# ax_delta[1].grid()
#
# ax_delta[0].plot(t_red_x, red_data[0][12:]-data[0][12:])
# ax_delta[1].plot(t_x, red_data[1][12:]-data[1][12:])
#
# ax_delta[0].set(xlabel='frame', ylabel='delta x',
#        title='Delta X Position Between Markers')
#
# ax_delta[1].set(xlabel='frame', ylabel='delta y',
#        title='Delta Y Position Between Markers')
#
# plt.show()
#
