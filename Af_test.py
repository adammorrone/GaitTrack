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
    all_pos_established = False


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


def last_known_value(arr):
    value_found = False
    k = len(arr) - 1
    while not value_found and k > 0:
        if arr[k]:
            break
        k = k - 1

    return arr[k]


def drawline(color1, id1, color2, id2, b=255, g=255, r=255, thickness=2):
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

    cv2.line(frame, (x1, y1), (x2, y2), (b, g, r), thickness)


# list of Color objects to look for
colors = [Color("G", (23, 65, 19), (38, 248, 119), 1)
          ]

ref_ID = 0

# Color("G", (40, 40, 30), (101, 255, 255), 4)]


    # [Color("G", (54, 36, 68), (90, 180, 161), 2),
    #      Color("R", (118, 80, 96), (205, 201, 234), 1),
    #      Color("Y", (24, 65, 110), (44, 137, 255), 1),
    #      Color("B", (110, 20, 0), (255, 255, 131), 1)]

angles = []

total_objects = 0
for color in colors:
    color.x_pos = [[]] * color.num_objects
    color.y_pos = [[]] * color.num_objects
    total_objects = total_objects + color.num_objects


vs = cv2.VideoCapture(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Nitinol_Implant\Af_tests\500C_15min.mp4')

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
    frame = imutils.resize(frame, width=300)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask

    for color in colors:
        mask = cv2.inRange(hsv, color.lower_bound_HSV, color.upper_bound_HSV)

        # mask = cv2.erode(mask, None, iterations=1)
        # mask = cv2.dilate(mask, None, iterations=6)
        # mask = cv2.erode(mask, None, iterations=5)

        cv2.imshow("Frame", mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
    #     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    #
    #     cnts = imutils.grab_contours(cnts)
    #
    #     center = None
    #
    #     # sort contours by y position, then use
    #     # them to compute the minimum enclosing circle and
    #     # centroid
    #
    #     ID = -1
    #     for i in range(0, len(cnts)):
    #
    #         ID = ID + 1
    #
    #         ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
    #         M = cv2.moments(cnts[i])
    #         # draw the circle and centroid on the frame,
    #         # then update the list of tracked points
    #
    #         x = int(M["m10"] / M["m00"])
    #         y = int(M["m01"] / M["m00"])
    #
    #         cv2.circle(frame, (int(x), int(y)), int(radius),
    #                    (255, 0, 0), 2)
    #
    #         correct_ID = False
    #
    #         # tracks dots through discontinuity
    #
    #         if len(cnts) == color.num_objects:
    #             color.all_pos_established = True
    #
    #         elif color.all_pos_established:
    #             multiplier = 0
    #             while not correct_ID and multiplier < 20:
    #                 for j in range(ID, color.num_objects):
    #                     minimum = min(m for m in color.y_pos[j] if m is not None)
    #                     low = minimum * (1 - multiplier*0.01)
    #                     maximum = max(m for m in color.y_pos[j] if m is not None)
    #                     high = maximum * (1 + multiplier*0.01)
    #                     if low <= y <= high:
    #                         ID = j
    #                         correct_ID = True
    #                         break
    #                 multiplier = multiplier + 1
    #
    #         if color.all_pos_established:
    #             cv2.putText(frame, color.label + str(ID + 1), (int(x), int(y - radius)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    #
    #             # hard coded green as marker color!
    #             if color.label != 'G' and len(colors[ref_ID].x_pos[0]) and len(color.x_pos[ID]):
    #
    #                 x1 = int(sum(colors[ref_ID].x_pos[0]) / len(colors[ref_ID].x_pos[0]))
    #                 x2 = int(sum(colors[ref_ID].x_pos[1]) / len(colors[ref_ID].x_pos[0]))
    #                 y1 = int(sum(colors[ref_ID].y_pos[0]) / len(colors[ref_ID].y_pos[0]))
    #                 y2 = int(sum(colors[ref_ID].y_pos[1]) / len(colors[ref_ID].y_pos[0]))
    #
    #                 rise = y2 - y1
    #                 run = x2 - x1
    #                 line_slope = -run / rise
    #                 offset_left = y1 - x1 * line_slope
    #                 offset_right = y2 - x2 * line_slope
    #
    #                 left_boundary = int((sum(colors[ref_ID].y_pos[0])/len(colors[ref_ID].y_pos[0]) - offset_left) / line_slope)
    #                 if last_known_value(color.x_pos[ID]) < left_boundary <= x:
    #                     for k in range(len(color.x_pos[ID])):
    #                         color.x_pos[ID][k] = None
    #                         color.y_pos[ID][k] = None
    #
    #                 right_boundary = int((sum(colors[ref_ID].y_pos[1])/len(colors[ref_ID].y_pos[1]) - offset_right) / line_slope)
    #                 if x >= right_boundary:
    #                     continue
    #
    #             color.x_pos[ID] = color.x_pos[ID] + [int(x)]
    #             color.y_pos[ID] = color.y_pos[ID] + [int(y)]
    #
    #         if ID == color.num_objects-1:
    #             break
    #
    #     # fills in any holes to keep the array lengths in sync with the frame counts
    #     for i in range(color.num_objects):
    #         if len(color.x_pos[i]) < frame_count:
    #             color.x_pos[i] = color.x_pos[i] + [None]
    #             color.y_pos[i] = color.y_pos[i] + [None]
    #
    # for color in colors:
    #     for obj in range(color.num_objects):
    #         for i in range(1, len(color.x_pos[obj])):
    #             # if either of the tracked points are None, ignore them
    #
    #             if color.x_pos[obj][i - 1] is None or color.x_pos[obj][i] is None:
    #                 continue
    #
    #             # draw the trace lines
    #             cv2.line(frame, (color.x_pos[obj][i - 1], color.y_pos[obj][i - 1]),
    #                      (color.x_pos[obj][i], color.y_pos[obj][i]), (4, 236, 255), 1)

    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    #
    # if the 'q' key is pressed, stop the loop

    if key == ord("q"):
        break

vs.release()
