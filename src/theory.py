import copy
import time
import warnings

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


def sqrt0(x):
    return np.sqrt(x * (x > 0))


def g1(m, l, s, d, h, gamma, t):
    return max0((m * np.sin(gamma + np.arctan(h/d)) - sqrt0(m**2 * np.sin(gamma + np.arctan(h/d))**2 + s**2 - m**2)) / np.sqrt(1 + h**2/d**2))


def g2(m, l, s, d, h, gamma, t):
    return max0((m * np.sin(gamma + np.arctan(h/d)) + sqrt0(m**2 * np.sin(gamma + np.arctan(h/d))**2 + s**2 - m**2)) / np.sqrt(1 + h**2/d**2))


def g3(m, l, s, d, h, gamma, t):
    return max0((m * np.sin(gamma + np.arctan(h/d)) + sqrt0(m**2 * np.sin(gamma + np.arctan(h/d))**2 + l**2 - m**2)) / np.sqrt(1 + h**2/d**2))


def f_tilde(m, l, s, d, h, gamma, t):
    return g3(m, l, s, d, h, gamma, t) - g2(m, l, s, d, h, gamma, t) + g1(m, l, s, d, h, gamma, t)


def t_minus(m, l, s, d, h, gamma, t):
    return max0(minn(g3(m, l, s, d, h, gamma, t), m * np.sin(gamma) - t))


def t_plus(m, l, s, d, h, gamma, t):
    return max0(minn(g3(m, l, s, d, h, gamma, t), m * np.sin(gamma) + t))


def f(m, l, s, d, h, gamma, t):
    t_minus_val = t_minus(m, l, s, d, h, gamma, t)
    t_plus_val = t_plus(m, l, s, d, h, gamma, t)
    return max0(minn(g1(m, l, s, d, h, gamma, t), t_plus_val) - t_minus_val) + max0(minn(t_plus_val, g3(m, l, s, d, h, gamma, t)) - maxx(g2(m, l, s, d, h, gamma, t), t_minus_val))


def find_max(func, params: dict):
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

    return maxi, params


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


if __name__ == '__main__':

    # Beispielwerte für die Variablen
    l = .63  # maximale Akzeptanzlänge
    s = .5   # kürzeste Akzeptanzlänge
    m = .5   # tatsächliche Länge der Verbindung
    gamma = np.pi / 20  # Neigungswinkel
    d = 3.   # waagerechter Kameraabstand
    h = .5   # vertikaler Kameraabstand
    t = np.inf  # maximale Tiefendifferenz

    params = {
        "l": l,
        "s": s,
        "m": [s, l, 100],
        "gamma": (-np.pi/2, np.pi*3/2, 100),
        "d": [1., 5., 100],
        "h": (-1.5, 1.5, 100),
        "t": t,
    }

    params2 = {
        "l": l,
        "s": s,
        "m": m,
        "gamma": "x",
        "d": d,
        "h": "y",
        "t": t,
    }

    plot(func=f, func_label="f", params=params2, range_x=(-np.pi/2, np.pi*3/2), range_y=(-1.5, 3), points_per_axis=100)

    """for m in np.arange(m,l,0.01):
        params["m"] = float(m)
        # maxi = find_max(func=f, params=params, range_x=(-np.pi/2, np.pi*3/2), range_y=(-1.5, 3), points_per_axis=10000)
        maxi = find_max(func=f, params=params)
        print(f"m = {m} => max = {maxi}")"""
    # print(find_max(func=f, params=params))
    #test, maxi = find_max(func=f, params=params)
    # print(f_tilde(**{'l': 0.63, 's': 0.5, 'm': 0.63, 'gamma': 0.6505318121069774, 'd': 4.3939393939393945, 'h': 0.015151515151515194, 't': np.inf}))
    #print(maxi)
    """
    m = 0.5 => max = 0.3832726126710673
    m = 0.51 => max = 0.48376557584662094
    m = 0.52 => max = 0.5260960171238898
    m = 0.53 => max = 0.5590540230803155
    m = 0.54 => max = 0.5872261859649304
    m = 0.55 => max = 0.6124027578348695
    m = 0.56 => max = 0.6354583130171356
    m = 0.57 => max = 0.6569455525679014
    m = 0.58 => max = 0.6772008538645009
    m = 0.59 => max = 0.6964749168486538
    m = 0.6 => max = 0.7149257821666968
    m = 0.61 => max = 0.7326943627039589
    m = 0.62 => max = 0.7498766336281268
    
    (0.76502622, {'l': 0.63, 's': 0.5, 'm': 0.63, 'gamma': 0.7139983303613167, 'd': 3.0202020202020203, 'h': -0.18181818181818177, 't': inf})
    """
