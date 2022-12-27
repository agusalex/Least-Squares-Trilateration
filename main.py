import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy import stats

from easy_trilateration.kalman import KalmanFilter
from easy_trilateration.model import *
import argparse
from easy_trilateration.least_squares import *
from easy_trilateration.graph import *


def trilateration_example():
    arr = [Circle(100, 100, 50),
           Circle(100, 50, 50),
           Circle(50, 50, 50),
           Circle(50, 100, 50)]
    result, meta = easy_least_squares(arr)
    create_circle(result, target=True)
    draw(arr)


def history_example() -> [Trilateration]:
    arr = Trilateration([Circle(100, 100, 70.71),
                         Circle(100, 50, 50),
                         Circle(50, 50, 0), Circle(50, 100, 50)])

    arr2 = Trilateration([Circle(100, 100, 50),
                          Circle(100, 50, 70.71),
                          Circle(50, 50, 50),
                          Circle(50, 100, 0)])

    arr3 = Trilateration([Circle(100, 100, 0),
                          Circle(100, 50, 50),
                          Circle(50, 50, 70.71),
                          Circle(50, 100, 50)])

    arr4 = Trilateration([Circle(100, 100, 50),
                          Circle(100, 50, 0),
                          Circle(50, 50, 50),
                          Circle(50, 100, 70.71)])

    hist: [Trilateration] = [arr, arr2, arr3, arr4, arr]

    solve_history(hist)

    _a = animate(hist)
    return _a


def hasOutliers(check):
    if not check:
        return False
    elif len(check) > 0:
        if isinstance(check[0], ndarray):
            if check[0].size > 0:
                if check[0][0] != 0:
                    return True
    return False


def get_outliers(threshold, data):
    ret = []
    for column_index, column_name in enumerate(data):
        if column_index > 0:
            z = np.abs(stats.zscore(data[column_name]))
            filtered = np.where(z > threshold)
            outliers = list(filtered)
            if z.size > 1 and hasOutliers(filtered):
                for outlier in outliers:
                    ret.append((column_index, outlier))
    return ret


def filter_outliers_median(threshold, data):
    outliers_set = set()
    if threshold is not None:
        for column_index, outliers in get_outliers(threshold, data):
            for outlier in outliers:
                outliers_set.add(outlier)
    for item in outliers_set:
        data.drop(item, inplace=True)
    return data


def amountOfDifferentNodes(circles: [Circle]):
    diffnodes = set()
    for circle in circles:
        diffnodes.add("x:" + str(circle.center.x) + "y:" + str(circle.center.y))
    return len(diffnodes)


def distance(row_i):
    return rssi_to_distance(row_i['rssi']) + float(node_location[row['node']]['offset'])


def rssi_to_distance(rssi, a=36, n=27):
    return 10 ** (-1 * (rssi + a) / n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trilateration solver and 2D grapher')  # cuadrado_diagonal_kalman.csv
    parser.add_argument('--file', nargs='?', help='data filename', default='resources/simulation/capture_combined.csv')

    args = parser.parse_args()

    _filename = args.file

    df = pd.read_csv(_filename)

    temp_tril = []
    history = []

    node = dict()

    # draws = []
    #  for value in node.values():
    #      draws.append(create_point(value))
    #  draw(draws)
    node_location = {
        "192.168.4.3": {"x": "0",
                        "y": "0",
                        "offset": "0",
                        "C": "0",
                        "R": "0"}
        ,
        "192.168.4.5": {"x": "2.5",
                        "y": "0",
                        "offset": "2.25",
                        "C": "0",
                        "R": "0"},
        "192.168.4.6": {"x": "5",
                        "y": "0",
                        "offset": "0",
                        "C": "0",
                        "R": "0"},
        "192.168.4.7": {"x": "0",
                        "y": "2.5",
                        "offset": "0.5",
                        "C": "0",
                        "R": "0"},
        "192.168.4.8": {"x": "5",
                        "y": "2.5",
                        "offset": "0",
                        "C": "0",
                        "R": "0"},
        "192.168.4.10": {"x": "0",
                         "y": "5",
                         "offset": "0",
                         "C": "0",
                         "R": "0"},
        "192.168.4.11": {"x": "2.5",
                         "y": "5",
                         "offset": "-2.25",
                         "C": "0",
                         "R": "0"},
        "192.168.4.12": {"x": "5",
                         "y": "5",
                         "offset": "0",
                         "C": "0",
                         "R": "0"}
    }

    actual = []

    # df.sort_values('millis', inplace=True)

    # filtered = filter_outliers_median(1.7, df)

    df.sort_values('millis', inplace=True)
    df['millis'] = df.apply(lambda x: round(x.millis / 1000), axis=1)
    millis = df['millis'][0]
    # group_millis = file.groupby['millis']
    group_by_node = df.groupby(['node'])

    convert = group_by_node.filter(lambda x: True)
    group_by_millis = convert.groupby(['millis'])
    group_by_millis_filter = group_by_millis.filter(lambda x: len(pd.unique(x['node'])) >= 3)

    for name, group in group_by_millis_filter.groupby(['millis']):
        for index, row in group.iterrows():
            if row['node'] in node_location.keys():
                temp_tril.append(
                    Circle(float(node_location[row['node']]['x']), float(node_location[row['node']]['y']),
                           distance(row)))
            elif 'x' in row:
                temp_tril.append(
                    Circle(float(row['x']), float(row['y']),
                           rssi_to_distance(row['rssi'])))
                actual.append((row['target_x'], row['target_y']))
        history.append(Trilateration(temp_tril.copy()))
        temp_tril = []

    solve_history(history)
    _a = static(history, actual)
