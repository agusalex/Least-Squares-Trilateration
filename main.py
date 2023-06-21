import json

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy import stats
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
    if len(node_location) > 0:
        n = node_location[row['node']]
        return rssi_to_distance(row_i['rssi'], int(n["A"]), int(n["N"])) + float(n['offset'])
    else:
        return rssi_to_distance(row_i['rssi'])


def rssi_to_distance(rssi, a=36, n=27):
    return 10 ** (-1 * (rssi + a) / n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trilateration solver and 2D grapher')  # cuadrado_diagonal_kalman.csv
    parser.add_argument('--file', nargs='?', help='data filename',
                        default='resources/simulation/capture_combined.csv')

    args = parser.parse_args()

    _filename = args.file
    path_groups = _filename.split("/")
    default_simulation = {"actual": [], "min_nodes": 5, "node_location": {
    }}
    try:
        with open("/".join(_filename.split("/")[:len(path_groups) - 1]) + "/locations.json") as json_f:
            json_file = json.load(json_f)
            file = path_groups[len(path_groups) - 1]
            csv_metadata = json_file.get(file, default_simulation)
            node_location = json_file["node_location"]
            min_nodes = csv_metadata["min_nodes"]
            actual = csv_metadata["actual"]
            print(csv_metadata)
    except FileNotFoundError:
        min_nodes = 0
        node_location = {}
        actual = []
    df = pd.read_csv(_filename)

    temp_tril = []
    history = []

    node = dict()

    df.sort_values('millis', inplace=True)
    millis = df['millis'][0]
    group_by_node = df.groupby(['node'])
    convert = group_by_node.filter(lambda x: True)
    group_by_millis = convert.groupby(['millis'])
    group_by_millis_filter = group_by_millis.filter(lambda x: len(pd.unique(x['node'])) >= min_nodes)

    for name, group in group_by_millis_filter.groupby(['millis']):
        for index, row in group.iterrows():
            if row['node'] in node_location.keys():
                temp_tril.append(
                    Circle(float(node_location[row['node']]['x']), float(node_location[row['node']]['y']),
                           distance(row)))
            elif 'x' in row:
                temp_tril.append(
                    Circle(float(row['x']), float(row['y']),
                           distance(row)))
                actual.append((row['target_x'], row['target_y']))

        history.append(Trilateration(temp_tril.copy()))
        temp_tril = []

    solve_history(history)
    _a = static(history, actual)
