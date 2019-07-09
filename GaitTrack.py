# -- coding: cp1252 --

import cv2
import imutils
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import pickle
import csv
import os

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
        if arr[k] is not None:
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



ref_I_D = 0

# Color("G", (40, 40, 30), (101, 255, 255), 4)]


    # [Color("G", (54, 36, 68), (90, 180, 161), 2),
    #      Color("R", (118, 80, 96), (205, 201, 234), 1),
    #      Color("Y", (24, 65, 110), (44, 137, 255), 1),
    #      Color("B", (110, 20, 0), (255, 255, 131), 1)]

angles = []

directory = r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\obstruction_footage_white'

for file_name in os.listdir(directory):

    file_name = '2-3_a_24.mp4'

    colors = [Color("G", (50, 101, 56), (97, 255, 255), 2),
              Color("R", (159, 100, 51), (186, 255, 255), 4),
              Color("B", (96, 71, 20), (141, 255, 255), 4)
              ]

    colors_top_light = [Color("G", (54, 101, 49), (72, 255, 255), 2),
                        Color("R", (159, 100, 51), (186, 255, 255), 4),
                        Color("B", (100, 56, 0), (116, 255, 255), 4)
                        ]

    # colors = colors_top_light

    total_objects = 0
    for color in colors:
        color.x_pos = [[]] * color.num_objects
        color.y_pos = [[]] * color.num_objects
        total_objects = total_objects + color.num_objects

    vs = cv2.VideoCapture(directory + r'\\' + file_name)

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
        frame = imutils.resize(frame, width=1000)

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask

        for color in colors:
            mask = cv2.inRange(hsv, color.lower_bound_HSV, color.upper_bound_HSV)

            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=6)
            mask = cv2.erode(mask, None, iterations=5)

            if color.label == 'O':
                cv2.imshow("Frame", mask)

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


            I_D = -1
            if I_D >= 4:
                pass

            for i in range(0, len(cnts)):
                if I_D >= 4:
                    pass

                I_D = I_D + 1

                if I_D >= 4:
                    pass

                ((x, y), radius) = cv2.minEnclosingCircle(cnts[i])
                M = cv2.moments(cnts[i])
                # draw the circle and centroid on the frame,
                # then update the list of tracked points

                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])

                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (255, 0, 0), 2)

                correct_I_D = False

                # tracks dots through discontinuity

                if len(cnts) == color.num_objects:
                    color.all_pos_established = True

                elif color.all_pos_established:
                    multiplier = 0
                    while not correct_I_D and multiplier < 20:
                        for j in range(I_D, color.num_objects):
                            if I_D >= 4:
                                pass
                            minimum = min(m for m in color.y_pos[j] if m is not None)
                            low = minimum * (1 - multiplier*0.01)
                            maximum = max(m for m in color.y_pos[j] if m is not None)
                            high = maximum * (1 + multiplier*0.01)
                            if low <= y <= high:
                                if I_D >= 4:
                                    pass
                                I_D = j
                                if I_D >= 4:
                                    pass
                                correct_I_D = True
                                if I_D >= 4:
                                    pass
                                break
                        if I_D >= 4:
                            pass
                        multiplier = multiplier + 1
                        if I_D > 3:
                            pass
                        if I_D == 4:
                            pass






                if color.all_pos_established:
                    cv2.putText(frame, color.label + str(I_D + 1), (int(x), int(y - radius)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    if I_D >= 4:
                        pass

                    # hard coded green as marker color!
                    if color.label != 'G' and len(colors[ref_I_D].x_pos[0]) and len(color.x_pos[I_D]):

                        x1 = int(sum(c for c in colors[ref_I_D].x_pos[0] if c is not None) / len(colors[ref_I_D].x_pos[0]))
                        x2 = int(sum(c for c in colors[ref_I_D].x_pos[1] if c is not None) / len(colors[ref_I_D].x_pos[0]))
                        y1 = int(sum(c for c in colors[ref_I_D].y_pos[0] if c is not None) / len(colors[ref_I_D].y_pos[0]))
                        y2 = int(sum(c for c in colors[ref_I_D].y_pos[1] if c is not None) / len(colors[ref_I_D].y_pos[0]))

                        if I_D >= 4:
                            pass

                        rise = y2 - y1
                        run = x2 - x1
                        if rise != 0:
                            line_slope = -run / rise
                        else:
                            line_slope = 0.0000001
                        offset_left = y1 - x1 * line_slope
                        offset_right = y2 - x2 * line_slope

                        if I_D >= 4:
                            pass

                        left_boundary = int((sum(c for c in colors[ref_I_D].y_pos[0] if c is not None)/len(colors[ref_I_D].y_pos[0]) - offset_left) / line_slope)
                        last = last_known_value(color.x_pos[I_D])
                        if last is not None and last < left_boundary <= x:
                            for k in range(len(color.x_pos[I_D])):
                                color.x_pos[I_D][k] = None
                                color.y_pos[I_D][k] = None
                            if I_D >= 4:
                                pass


                        right_boundary = int((sum(c for c in colors[ref_I_D].y_pos[1] if c is not None)/len(colors[ref_I_D].y_pos[1]) - offset_right) / line_slope)
                        if x >= right_boundary:
                            continue

                    if I_D >= 4:
                        pass

                    # if len(color.y_pos[I_D]) < 50 or not color.all_pos_established or abs(int(y) - last_known_value(color.y_pos[I_D])) < 80:
                    color.x_pos[I_D] = color.x_pos[I_D] + [int(x)]
                    color.y_pos[I_D] = color.y_pos[I_D] + [int(y)]

                if I_D >= 4:
                    pass

                if I_D == color.num_objects-1:
                    break

            # fills in any holes to keep the array lengths in sync with the frame counts
            for i in range(color.num_objects):
                if len(color.x_pos[i]) < frame_count:
                    color.x_pos[i] = color.x_pos[i] + [None]
                    color.y_pos[i] = color.y_pos[i] + [None]

        for color in colors:
            for obj in range(color.num_objects):
                for i in range(1, len(color.x_pos[obj])):
                    # if either of the tracked points are None, ignore them

                    if color.x_pos[obj][i - 1] is None or color.x_pos[obj][i] is None:
                        continue

                    # draw the trace lines
                    cv2.line(frame, (color.x_pos[obj][i - 1], color.y_pos[obj][i - 1]),
                             (color.x_pos[obj][i], color.y_pos[obj][i]), (4, 236, 255), 1)

            # drawline(colors[2], 0, colors[1], 0)
            # drawline(colors[2], 1, colors[1], 1)
            # drawline(colors[2], 2, colors[1], 2)
            # drawline(colors[2], 3, colors[1], 3)

            # draws containing lines

            # x1 = int(sum(colors[ref_I_D].x_pos[0])/len(colors[ref_I_D].x_pos[0]))
            # x2 = int(sum(colors[ref_I_D].x_pos[1])/len(colors[ref_I_D].x_pos[0]))
            # y1 = int(sum(colors[ref_I_D].y_pos[0])/len(colors[ref_I_D].y_pos[0]))
            # y2 = int(sum(colors[ref_I_D].y_pos[1])/len(colors[ref_I_D].y_pos[0]))
            #
            # rise = y1 - y2
            # run = x1 - x2
            #
            # x3 = int(x1 + rise * 1.5)
            # x4 = int(x2 - rise * 1.5)
            # y3 = int(y1 - run * 1.5)
            # y4 = int(y2 + run * 1.5)
            #
            # x1 = int(x1 - rise * 1.5)
            # x2 = int(x2 + rise * 1.5)
            # y1 = int(y1 + run * 1.5)
            # y2 = int(y2 - run * 1.5)
            #
            # cv2.line(frame, (x1, y1), (x3, y3), (0, 0, 255), 2)
            # cv2.line(frame, (x2, y2), (x4, y4), (0, 0, 255), 2)
            #
            # drawline(colors[ref_I_D], 0, colors[ref_I_D], 1, 0, 0, 255)

            # cv2.line(frame, (int(sum(colors[1].x_pos[0])/len(colors[1].x_pos[0])), (int(sum(colors[1].y_pos[0])/len(colors[1].y_pos[0])))),
            #          (int(sum(colors[1].x_pos[0])/len(colors[1].x_pos[0])), int(sum(colors[1].y_pos[0])/len(colors[1].y_pos[0]))), (0, 0, 255), 2)
            #
            # cv2.line(frame, (int(sum(colors[1].x_pos[1]) / len(colors[1].x_pos[1])), int(sum(colors[1].y_pos[1]) / len(colors[1].y_pos[1]))),
            #          (int(sum(colors[1].x_pos[1]) / len(colors[1].x_pos[1])), int(sum(colors[1].y_pos[1]) / len(colors[1].y_pos[1]))), (0, 0, 255), 2)




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

    with open(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\data\\' + file_name[:-4] + '.p', 'wb') as fp:
        pickle.dump(colors, fp)

