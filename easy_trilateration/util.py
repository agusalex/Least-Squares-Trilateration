import copy
import statistics

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from easy_trilateration.kalman import *


def median(step):
    return [statistics.median(step)]


def mean(step):
    return [statistics.mean(step)]


def median_kalman(step):
    return [statistics.median(KalmanFilter().kalman_filter(step))]


def kalman(data):
    return KalmanFilter().kalman_filter(data)


def function_per_step(file, function):
    signal_in_steps = cut_signal_in_steps(file['rssi'].values, find_step_cuts(file.values))
    filtered = []
    for step in signal_in_steps:
        for filtered_step in function(step):
            filtered.append(filtered_step)
    return filtered


def cut_signal_in_steps(signal, cuts):
    signals_per_step = []
    for i in range(len(cuts) - 1):
        origin = int(cuts[i])
        to = int(cuts[i + 1])
        signals_per_step.append(signal[origin:to])

    signals_per_step.append(signal[cuts[len(cuts) - 1]:len(signal)])
    return signals_per_step


def find_step_cuts(file):
    cuttings = [0]
    index = file[0][0]
    cut = 0
    for step, signal in file:
        if step != index:
            cuttings.append(cut)
            index = step
        cut += 1
    return cuttings


def find_steps(distance, meters_per_step=1):
    steps = [distance[0]]
    index = distance[0]
    for step in distance:
        if step != index:
            steps.append(step * meters_per_step)
            index = step
    return steps


def find_coeficient(distance, signal):
    C, residual = fit(distance_to_rssi, np.array(distance), np.array(signal))
    return round(C.item()), round(100 - residual)


def log_fit(distance, c):
    return distance_to_rssi(distance, c)


def linear_fit(signal):
    m, b = np.polyfit(range(0, len(signal)), signal, 1)
    return m * range(0, len(signal)) + b


def find_coeficient_adaptive(distance, signal, C=35.5510920, N=29.0735592, B=11.8099735):
    initial_guess = dict(c=C, n=N, b=B)
    print(int(C), int(N), int(B))

    regressor = lmfit.Model(distance_to_rssi_adaptive)
    results = regressor.fit(signal, x=np.array(distance), **initial_guess, method="slsqp")
    residual = np.linalg.norm(signal - distance_to_rssi_adaptive(distance,
                                                                 results.values['c'],
                                                                 results.values['n'],
                                                                 results.values['b']))
    print("R = " + str(100 - residual) + " Result" + str(results.values))
    if results.values['b'] < 2:
        return results.values['c'], results.values['n'], 2, 100 - residual
    return results.values['c'], results.values['n'], results.values['b'], 100 - residual


def log_fit_adaptive(distance, c, n, b):
    return distance_to_rssi_adaptive(distance, c, n, b)


# objective function
def distance_to_rssi(x, c):
    return c - 20 * np.log10(4 * np.pi * x)


def distance_to_rssi_adaptive(x, c, n, b):
    if b < 2:
        return - n * (np.log10(x) / np.log10(2)) - c
    return - n * (np.log10(x) / np.log10(b)) - c


def rssi_to_distance_adaptive(x_values, c, n, b):
    y_values = []
    for x in x_values:
        y_values.append(b ** (-1 * (x + c) / n))
    return y_values


def rssi_to_distance(x_values, c):
    y_values = []
    for x in x_values:
        y_values.append((2 ** ((-x + c - 40) / 20) * 5 ** (-(x - c) / 20)) / np.pi)
    return y_values


def fit(func, x_values, y_values):
    popt, pcov = curve_fit(func, x_values, y_values)
    residual = np.linalg.norm(y_values - func(x_values, *popt))
    return popt, residual


def median_filter(data, w_size=13):
    """ Median filter for 1d data. Edges are handled with a window shift. """
    data2 = copy.deepcopy(data)
    windowed_data = window(data2, w_size)
    w_half_size = w_size // 2
    for i, w_data in enumerate(windowed_data):
        w_data_sort = sorted(w_data)
        data2[i] = w_data_sort[w_half_size]  # zero indexing handles shift to middle
    return data2


def window(data, w_size):
    """ Get all windows for a list with an odd window size. """
    if not getattr(data, '__iter__', False):
        raise ValueError("data must be an iterable")
    if w_size % 2 != 1:
        raise ValueError("window size must be odd")
    if w_size > len(data):
        raise ValueError("window size must be less than the data size")
    w_half_size = w_size // 2
    windows = []
    for i, _ in enumerate(data):
        if i < w_half_size:
            win = data[0:w_size]
            windows.append(win)
            continue
        if i > len(data) - 1 - w_half_size:
            win = data[-w_size:]
            windows.append(win)
            continue
        win = data[i - w_half_size:i + w_half_size + 1]
        windows.append(win)
    return windows


def plot_signals(yi, labels, ylabel="Y", xlabel="X", title="RSSI over Distance", xi=None, ):
    """

    Auxiliary function for plotting

    input:
        - labels: labels of input signals

    output:
        - display plot

    """

    if xi is None:
        xi = range(len(yi[0]))
    alphas = [1, 0.45, 0.45, 0.45, 0.45]  # just some opacity values to facilitate visualization

    plt.figure()

    for j, y in enumerate(yi):  # iterates on all signals
        plt.plot(xi, y, '-o', label=labels[j], markersize=2, alpha=alphas[j])

    plt.grid()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()

    return
