import copy
import os
import time
import warnings

import pandas as pd

from src.analyzer import cl

use_numpy = True

if use_numpy:
    warnings.warn("Install cupy for parallelized calculation")
    import numpy as np
else:
    try:
        import cupy as np
        import numpy
        use_numpy = False
    except ImportError:
        warnings.warn("Install cupy for parallelized calculation")
        import numpy as np
        use_numpy = True

import matplotlib
matplotlib.use('TkAgg')  # oder ein anderes fensterbasiertes Backend wie 'Qt5Agg'
import matplotlib.pyplot as plt
plt.ion()  # Schaltet den interaktiven Modus ein
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display


# Function to format the ticks as multiples of pi
def format_func_rad(value, tick_number):
    N = int(np.round(2*value / np.pi))
    if N == 0:
        return "0"
    elif N == 2:
        return r"$\pi$"
    elif N == -2:
        return r"-$\pi$"
    elif N == 1:
        return r"$\pi$/2"
    elif N == -1:
        return r"-$\pi$/2"
    else:
        return r"${0}\pi$".format(N)

# Definition der Formeln

def maxx(a,b):
    return np.max(np.array([a, b]), axis=0)


def minn(a,b):
    return np.min(np.array([a, b]), axis=0)


def max0(a):
    return np.max(np.array([np.zeros_like(a), a]), axis=0)

def betw0(a,d):
    return np.min([max0(a), np.ones_like(a)*d], axis=0)


def sqrt0(x):
    return np.sqrt(x * (x > 0))


def g0(m, l, s, d, h, gamma, t, k):
    return betw0((m * np.sin(gamma + np.arctan(h / d)) - sqrt0(m ** 2 * np.sin(gamma + np.arctan(h / d)) ** 2 + l ** 2 - m ** 2)) / np.sqrt(1 + h ** 2 / d ** 2), d)


def g1(m, l, s, d, h, gamma, t, k):
    return betw0((m * np.sin(gamma + np.arctan(h / d)) - sqrt0(m ** 2 * np.sin(gamma + np.arctan(h / d)) ** 2 + s ** 2 - m ** 2)) / np.sqrt(1 + h ** 2 / d ** 2), d)


def g2(m, l, s, d, h, gamma, t, k):
    return betw0((m * np.sin(gamma + np.arctan(h / d)) + sqrt0(m ** 2 * np.sin(gamma + np.arctan(h / d)) ** 2 + s ** 2 - m ** 2)) / np.sqrt(1 + h ** 2 / d ** 2), d)


def g3(m, l, s, d, h, gamma, t, k):
    return betw0((m * np.sin(gamma + np.arctan(h / d)) + sqrt0(m ** 2 * np.sin(gamma + np.arctan(h / d)) ** 2 + l ** 2 - m ** 2)) / np.sqrt(1 + h ** 2 / d ** 2), d)


def f_tilde(m, l, s, d, h, gamma, t, k):
    return g3(m, l, s, d, h, gamma, t, k) - g2(m, l, s, d, h, gamma, t, k) + g1(m, l, s, d, h, gamma, t, k) - g0(m, l, s, d, h, gamma, t, k)


def t_minus(m, l, s, d, h, gamma, t, k):
    return max0(minn(g3(m, l, s, d, h, gamma, t, k), m * np.sin(gamma) - t))


def t_plus(m, l, s, d, h, gamma, t, k):
    return max0(minn(g3(m, l, s, d, h, gamma, t, k), m * np.sin(gamma) + t))


def f(m, l, s, d, h, gamma, t, k):
    t_minus_val = t_minus(m, l, s, d, h, gamma, t, k)
    t_plus_val = t_plus(m, l, s, d, h, gamma, t, k)
    return max0(minn(g1(m, l, s, d, h, gamma, t, k), t_plus_val) - max(g0(m, l, s, d, h, gamma, t, k), t_minus_val)) + max0(minn(t_plus_val, g3(m, l, s, d, h, gamma, t, k)) - maxx(g2(m, l, s, d, h, gamma, t, k), t_minus_val))


def f_2(m, l, s, d, h, gamma, t, k):
    k = np.where(0 <= k <= d - m*np.sin(gamma), k, np.nan)

    m_prime = np.sqrt((k + m * np.sin(gamma))**2 + (m * np.cos(gamma) + k * (h - m * np.cos(gamma)) / (d - m * np.sin(gamma)))**2)

    sin_part = (m * np.sin(gamma) + k) / m_prime
    gamma_prime = np.where(m * np.cos(gamma) + k * (h - m * np.cos(gamma)) / (d - m * np.sin(gamma)) >= 0,
                           np.arcsin(sin_part),
                           np.pi - np.arcsin(sin_part))

    return f(m_prime, l, s, d, h, gamma_prime, t, k)


def calc_stats_numerically(func, params: dict, use_nan=False):
    params = params.copy()

    ranges = []
    for k, v in params.items():
        if not isinstance(v, float):
            assert len(v) == 3, "Provide start, end, number of points for each variable to constucts np.linspace"
            start, end, size = params[k]
            ranges.append(np.linspace(start, end, size))

    Xs = np.meshgrid(*ranges)
    for k, v in params.items():
        if not isinstance(v, float):
            params[k] = Xs.pop(0)

    Z = func(**params)
    Xmax = np.unravel_index(np.argmax(Z, axis=None), Z.shape)
    maxi = float(Z[Xmax])

    s = []
    for k, v in params.items():
        if not isinstance(v, float):
            params[k] = float(params[k][Xmax])
            s.append(f"{k}={params[k]}")

    mean, median, std = (np.nanmean, np.nanmedian, np.nanstd) if use_nan else (np.mean, np.median, np.std)

    return {
            "Max": maxi,
            "Parameters for Max": params,
            "Average": float(mean(Z)),
            "Median": float(median(Z)),
            "StdDev": float(std(Z))
            }


def plot(func, func_label: str, params: dict, range_x: tuple[float, float], range_y: tuple[float, float], points_per_axis: int = 1000):
    params = params.copy()
    x_range = np.linspace(range_x[0], range_x[1], points_per_axis)
    y_range = np.linspace(range_y[0], range_y[1], points_per_axis)
    X, Y = np.meshgrid(x_range, y_range)

    c1, c2 = 0, 0
    param_str = []
    x_in_rad, y_in_rad = False, False
    for k, v in params.items():
        if v == "x" or v == "X":
            params[k] = X
            x_name = k
            c1 += 1
            param_str.append(f"{k}")
            if k == "gamma":
                x_in_rad = True
        elif v == "y" or v == "Y":
            params[k] = Y
            y_name = k
            c2 += 1
            param_str.append(f"{k}")
            if k == "gamma":
                y_in_rad = True
        else:
            param_str.append(f"{k}={v}")
    if c1 != 1 or c2 != 1:
        raise ValueError("Assign exactly one X and exactly one Y to the parameter you would like to plot.")
    param_str = ", ".join(param_str)

    ti = time.time()
    Z = func(**params)
    print(f" Calculation time: {round(time.time() - ti, 3)}s")
    print(f"Max is: {np.max(Z)}")

    # Erstellung des 3D-Plots
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if use_numpy:
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.contour3D(X, Y, Z, 20, cmap='gist_gray', linestyles="solid")

    else:
        X, Y, Z = X.get(), Y.get(), Z.get()
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        ax.contour3D(X, Y, Z, 20, cmap='gist_gray', linestyles="solid")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    if x_in_rad:
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/2))
        ax.xaxis.set_major_formatter(FuncFormatter(format_func_rad))
    if y_in_rad:
        ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/2))
        ax.xaxis.set_major_formatter(FuncFormatter(format_func_rad))

    # Beschriftung der Achsen
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(func_label)

    # Titel des Plots
    ax.set_title(f'{func_label}({param_str})')

    # Anzeige des Plots
    plt.show(block=True)


def show_statistics_for_multiple_connections(cons=("Nose-Neck", "Nose-LEye", "LEye-LEar", "Neck-LShoulder", "LShoulder-LElbow", "LElbow-LWrist", "Neck-MidHip", "MidHip-LHip", "LHip-LKnee"),
                                             print_console=True, save_path:str =None, load_path=None, func=f_2, params=None, m_points=50, gamma=(-np.pi/2, np.pi/2, 40), d=(1., 5., 40), h=(-1.5, 1.5, 40), k_points=40):

    assert all([con in cl.lengths_hr.keys() and con in cl.depth_deviations_hr.keys() for con in cons]), "Check your specified connections. Some are not in lengths_hr or depth_deviations_hr"

    if save_path is not None:
        if not (save_path.endswith(".csv") or save_path.endswith(".xlsx")):
            save_path += ".csv"
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Proceeding will overwrite this file.")
            print(f"Proceed? y/[n]")
            if input().lower() != 'y':
                return

    print("Starting...")

    if load_path is None:
        df = pd.DataFrame()
        dict_max_parameter = {}
        for con in cons:
            s, l = cl.lengths_hr[con]
            t = cl.depth_deviations_hr[con]
            t = t if t >= 0 else np.inf

            params["l"] = l
            params = {
                "l": l,
                "s": s,
                "m": (s, l, m_points),
                "gamma": gamma,
                "d": d,
                "h": h,
                "t": t,
                "k": (0, d-m*np.sin(gamma))
            }

            v = calc_stats_numerically(func=func, params=params)
            dict_max_parameter[con] = v.pop("Parameters for Max")
            df[con] = pd.Series(v)

            print(f"Connection {con} completed...")

        if print_console:
            print(df)
            print()
            print('\n'.join([f'{key}: {value}' for key, value in dict_max_parameter.items()]))

        if save_path is not None:
            if save_path.endswith(".csv"):
                df.to_csv(save_path)
            elif save_path.endswith(".xlsx"):
                df.to_excel(save_path)
            print(f"Saved to {save_path}")
    else:
        if load_path.endswith(".csv"):
            df = pd.read_csv(load_path, index_col=0)
        elif load_path.endswith(".xlsx"):
            df = pd.read_excel(load_path, index_col=0)
        else:
            raise ValueError("Load path is neither csv nor xlsx")


    # Set up the figure
    plt.figure(figsize=(12, 6))  # Adjusting size to accommodate around 10 connections

    # X-axis labels (Connections)
    x_labels = df.keys()
    x = range(len(x_labels))  # Numeric x-axis to position the dots

    # Plotting each metric
    plt.scatter(x, df.loc['Max'], color='red', label='Max')  # Max in red
    plt.scatter(x, df.loc['Average'], color='#00ff00', label='Average', alpha=1)  # Average in green
    plt.scatter(x, df.loc['Median'], color='blue', label='Median')  # Median in blue

    # Adding error bars for StdDev
    # The error bar is capped at 0 by calculating the lower limit: max(average - stddev, 0)
    stddev_values = df.loc['StdDev']
    average_values = df.loc['Average']
    lower_limits = [max(avg - std, 0) for avg, std in zip(average_values, stddev_values)]
    upper_limits = [avg + std for avg, std in zip(average_values, stddev_values)]
    lower_errors = average_values - lower_limits
    upper_errors = upper_limits - average_values
    errors = [lower_errors, upper_errors]
    plt.errorbar(x, average_values, yerr=errors, fmt='o', color='black', capsize=5, alpha=.7)

    # Labels and Title
    plt.xticks(x, x_labels, rotation=15)
    plt.ylabel('f in m')
    plt.title('Metrics for the theoretical total distance where validation fails')
    plt.legend()

    plt.grid(which='both', axis='y', linestyle='dotted', color='gray')
    plt.minorticks_on()  # Enable minor ticks for finer grid
    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.5)

    # Ensuring the y-axis starts at 0
    plt.ylim(bottom=0)

    # Show the plot
    plt.show(block=True)


# Beispielwerte für die Variablen
l = .63  # maximale Akzeptanzlänge
s = .5   # kürzeste Akzeptanzlänge
m = .58   # tatsächliche Länge der Verbindung
gamma = np.pi / 20  # Neigungswinkel
d = 3.   # waagerechter Kameraabstand
h = .5   # vertikaler Kameraabstand
t = 0.2  # maximale Tiefendifferenz
k = 0.

params_for_f = {
    "l": l,
    "s": s,
    "m": (s, l, 100),
    "gamma": (-np.pi/2, np.pi/2, 100),
    "d": (1., 5., 100),
    "h": (-1.5, 1.5, 100),
    "t": t,
    "k": k
}

params_for_f2 = {
    "l": l,
    "s": s,
    "m": (s, l, 10),
    "gamma": (-np.pi/2, np.pi/2, 10),
    "d": (1., 5., 10),
    "h": (-1.5, 1.5, 10),
    "t": t,
    "k": (0, d+m, 10)
}

params_for_view = {
    "m": m,
    "l": l,
    "s": s,
    "d": d,
    "h": "y",
    "gamma": "x",
    "t": t,
    "k": k
}

if __name__ == '__main__':



    # plot(func=f, func_label="f", params=params2, range_x=(-np.pi/2, np.pi*3/2), range_y=(-1.5, 5), points_per_axis=100)

    # show_statistics_for_multiple_connections(load_path="res/theoretical_stats_2.xlsx")

    """ti = time.time()
    print('\n'.join([f'{key}: {value}' for key, value in calc_stats_numerically(func=f_2, params=params_for_f, use_nan=True).items()]))
    print(time.time()-ti)"""

    print(f_2(l=0.63, s=0.5, m=0.6, gamma=-0.079, d=1.6, h=0.1, t=0.82, k=0.68))

    """p = {'l': 0.63, 's': 0.5, 'm': 0.6011111111111112, 'gamma': 0.5235987755982989, 'd': 1.0, 'h': -0.10606060606060602, 't': 0.2, 'k': 0.0}

    print(f"g_1: {g1(**p)}")
    print(f"g_2: {g2(**p)}")
    print(f"g_3: {g3(**p)}")
    print(f"t_plus: {t_plus(**p)}")
    print(f"t_min: {t_minus(**p)}")
    print(f"f: {f(**p)}")
    for k in range(10):
        k = k/10
        p["k"] =float(k)
        print()
        print(f"k: {k}")
        print(f"f_2 {f_2(**p)}")"""