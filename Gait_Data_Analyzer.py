import pickle
import os
from dataclasses import dataclass
import math
import csv


@dataclass
class Color:
    label: str
    lower_bound_HSV: tuple
    upper_bound_HSV: tuple
    num_objects: int = 1
    x_pos: list = None
    y_pos: list = None
    all_pos_established = False


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        # Only allow safe classes from builtins.
        if module == Color:
            return Color
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))


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


def convert_to_coordinates(color):
    o_x = int(sum([c for c in colors[0].x_pos[0] if c is not None]) /
              len([c for c in colors[0].x_pos[0] if c is not None]))

    o_y = int(sum([c for c in colors[0].y_pos[0] if c is not None]) /
              len([c for c in colors[0].y_pos[0] if c is not None]))

    x_coordinates = [[x - o_x if x is not None else x for x in color[2].x_pos[0]],
                     [x - o_x if x is not None else x for x in color[1].x_pos[0]],
                     [x - o_x if x is not None else x for x in color[2].x_pos[1]],
                     [x - o_x if x is not None else x for x in color[2].x_pos[1]],
                     [x - o_x if x is not None else x for x in color[1].x_pos[2]],
                     [x - o_x if x is not None else x for x in color[2].x_pos[2]],
                     [x - o_x if x is not None else x for x in color[1].x_pos[3]],
                     [x - o_x if x is not None else x for x in color[2].x_pos[3]]
                     ]

    y_coordinates = [[o_y - y if y is not None else y for y in color[2].y_pos[0]],
                     [o_y - y if y is not None else y for y in color[1].y_pos[0]],
                     [o_y - y if y is not None else y for y in color[2].y_pos[1]],
                     [o_y - y if y is not None else y for y in color[2].y_pos[1]],
                     [o_y - y if y is not None else y for y in color[1].y_pos[2]],
                     [o_y - y if y is not None else y for y in color[2].y_pos[2]],
                     [o_y - y if y is not None else y for y in color[1].y_pos[3]],
                     [o_y - y if y is not None else y for y in color[2].y_pos[3]]
                     ]

    return [x_coordinates, y_coordinates]


def slope(color1, id1, color2, id2):
    rise = color1.y_pos[id1][-1] - color2.y_pos[id2][-1]
    run = color1.x_pos[id1][-1] - color2.x_pos[id2][-1]
    return rise/run


directory = r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\data\obstruction_data_7-9-19'

for file_name in os.listdir(directory):

    pkl_file = open(directory + r'\\' + file_name, 'rb')

    colors = pickle.load(pkl_file)

    [x_coord, y_coord] = convert_to_coordinates(colors)

    tx = x_coord

    pkl_file.close()

    with open(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\data\coordinates_7-10-19\\' +
              file_name[:-2] + '.p', 'wb') as fp:
        pickle.dump(colors, fp)

    markers = ['B1', 'R1', 'B2', 'R2', 'B3', 'R3', 'B4', 'R4']

    for i in range(0, len(markers)):
        with open(r'C:\Users\amorrone\Google Drive\Colorado State\Research\Gait_Analysis\data\csv_coordinates_7-10-19\\'
                  + file_name[:-2] + '_' + markers[i] + '.csv', 'w', newline='') as myfile:

            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(y_coord[i])
