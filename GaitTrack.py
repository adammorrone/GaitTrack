# -- coding: cp1252 --

import cv2
import imutils
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math



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


# datatype should be two colors and the object number of intersst
# calcs sloep from two markers
# pnt1 should be higher than pnt2
def slope(color1, id1, color2, id2):
    rise = color1.y_pos[id1][-1] - color2.y_pos[id2][-1]
    run = color1.x_pos[id1][-1] - color2.x_pos[id2][-1]
    return rise/run


# calcs angle between two lines (4 markers)
# datatype should be for markers via color and object_id in descending y value
def angle(ln1_color1, id1_1, ln1_color2, id1_2, ln2_color1, id2_1, ln2_color2, id2_2):
    x1 = ln1_color1.x_pos[id1_1][-1]
    x2 = ln1_color2.x_pos[id1_2][-1]
    x3 = ln2_color1.x_pos[id2_1][-1]
    x4 = ln2_color2.x_pos[id2_2][-1]

    y1 = ln1_color1.y_pos[id1_1][-1]
    y2 = ln1_color2.y_pos[id1_2][-1]
    y3 = ln2_color1.y_pos[id2_1][-1]
    y4 = ln2_color2.y_pos[id2_2][-1]

    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x4, y3 - y4)

    theta = math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / ((math.hypot(x1 - x2, y1 - y2)) * math.hypot(x3 - x4, y3 - y4)))

    return math.degrees(theta)


def drawline(color1, id1, color2, id2):
    x1 = color1.x_pos[id1][-1]
    x2 = color2.x_pos[id2][-1]
    y1 = color1.y_pos[id1][-1]
    y2 = color2.y_pos[id2][-1]

    rise = y1 - y2
    run = x1 - x2

    x1 = int(x1 + run*1.5)
    x2 = int(x2 - run*1.5)
    y1 = int(y1 + rise*1.5)
    y2 = int(y2 - rise*1.5)

    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)


# list of Color objects to look for
colors = [Color("R", (0, 50, 20), (15, 255, 255), 4),
          Color("B", (121, 24, 14), (166, 125, 98), 4),
          Color("G", (52, 122, 14), (102, 255, 117), 2)]


        #  Color("G", (40, 40, 30), (101, 255, 255), 4)]


    # [Color("G", (54, 36, 68), (90, 180, 161), 2),
    #      Color("R", (118, 80, 96), (205, 201, 234), 1),
    #      Color("Y", (24, 65, 110), (44, 137, 255), 1),
    #      Color("B", (110, 20, 0), (255, 255, 131), 1)]

angles = []

total_objects = 0
for color in colors:
    color.x_pos = [[None]] * color.num_objects
    color.y_pos = [[None]] * color.num_objects
    total_objects = total_objects + color.num_objects


vs = cv2.VideoCapture(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\obstruction_footage\0_b_5.mp4')

# loop through frames
frame_count = 0
while True:
    frame_count = frame_count + 1

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

    for color in colors:
        mask = cv2.inRange(hsv, color.lower_bound_HSV, color.upper_bound_HSV)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=4)
        mask = cv2.erode(mask, None, iterations=3)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        center = None

        # sort contours by y position, then use
        # them to compute the minimum enclosing circle and
        # centroid

        if len(cnts) > color.num_objects:
            cnts.sort(key=lambda cnt: cv2.contourArea(cnt))
            cnts = cnts[0:color.num_objects]
        cnts.sort(key=lambda y_pos: cv2.moments(y_pos)['m01']/cv2.moments(y_pos)['m00'])

        ID = -1
        for i in range(0, len(cnts)):

            ID = ID + 1

            ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
            M = cv2.moments(cnts[i])
            # draw the circle and centroid on the frame,
            # then update the list of tracked points

            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            if frame_count > 0 and color.y_pos[i][-1]:
                if y > color.y_pos[i][-1] + 20 and ID < color.num_objects-1:
                    ID = ID + 1

            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)
            cv2.putText(frame, color.label + str(ID + 1), (int(x), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


            tempx = color.x_pos[ID].copy()
            tempy = color.y_pos[ID].copy()
            tempx.append(int(x))
            tempy.append(int(y))

            color.x_pos[ID] = tempx.copy()
            color.y_pos[ID] = tempy.copy()

            if ID == color.num_objects-1:
                break




    for color in colors:
        for obj in range(color.num_objects):
            for i in range(len(color.x_pos[obj])):
                # if either of the tracked points are None, ignore them

                if color.x_pos[obj][i - 1] is None or color.x_pos[obj][i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                cv2.line(frame, (color.x_pos[obj][i - 1], color.y_pos[obj][i - 1]),
                         (color.x_pos[obj][i], color.y_pos[obj][i]), (4, 236, 255), 1)

        # drawline(colors[0], 0, colors[1], 0)
        # drawline(colors[0], 1, colors[1], 1)
        # drawline(colors[0], 2, colors[1], 2)
        # drawline(colors[0], 3, colors[1], 3)

        # ang = int(angle(colors[0], 1, colors[0], 0, colors[3], 0, colors[0], 0))
        # cv2.putText(frame, str(ang), (colors[0].x_pos[1][-1]-20, colors[0].y_pos[1][-1]+20), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, (255, 255, 255), 2)
    #
    # angles.append(ang)

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
