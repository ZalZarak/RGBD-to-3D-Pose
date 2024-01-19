import copy
import os.path
import pickle
import statistics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore

from src.config import config
from src.perceptor import Perceptor

conf = config["Perceptor"]
conf["playback"] = True
conf["playback_file"] = " "
conf["countdown"] = 0
conf["save_joints"] = conf["save_bag"] = conf["save_performance"] = conf["show_rgb"] = conf["show_depth"] = conf["show_joints"] = conf["show_color_mask"] \
    = conf["simulate_limbs"] = conf["simulate_joints"] = conf["simulate_joint_connections"] = False

cl = Perceptor(**conf)

is_valid = lambda j: np.all(j != (0, 0, 0))


def read_joint_file(file_path):
    """
    Read joint file and return frames (joint positions), according times and number of frames
    """

    with open(file_path, "rb") as f:
        obj = list(pickle.load(f))

    frames = np.array(list(map(lambda f: f[1], obj)))
    times = np.array(list(map(lambda f: f[0], obj)))
    l = len(obj)

    return frames, times, l


def proportion_valid_joints(file_path):
    frames, times, l = read_joint_file(file_path)

    r = np.sum(np.all(frames != [0, 0, 0], axis=-1), axis=0).astype(int)
    return [[i / l] for i in r], [[i] for i in r]


def count_streak_times(file_path):
    frames, times, l = read_joint_file(file_path)

    res = [[] for _ in range(25)]
    on_streak = np.ones([25], dtype=float) * -1  # strike start time

    for t, frame in enumerate(frames[:-1]):
        for i, joint in enumerate(frame):
            if is_valid(joint):
                if on_streak[i] == -1:
                    on_streak[i] = times[t]
            else:
                if on_streak[i] >= 0:
                    res[i].append(times[t] - on_streak[i])
                    on_streak[i] = -1

    for i, joint in enumerate(frames[-1]):
        if on_streak[i] >= 0:
            res[i].append(times[-1] - on_streak[i])
            on_streak[i] = -1

    return res


def get_joint_speeds(file_path):
    frames, times, l = read_joint_file(file_path)

    res = [[] for _ in range(25)]

    for t, frame in enumerate(frames[1:]):
        t += 1
        for i, joint in enumerate(frame):
            if is_valid(joint) and is_valid(frames[t - 1, i]):
                ds = np.linalg.norm(joint - frames[t - 1, i])
                dt = times[t] - times[t - 1]
                res[i].append(ds / dt)
    return res


def get_invalid_connections(file_path):
    frames, times, l = read_joint_file(file_path)

    res_too_long = [[] for _ in range(len(cl.connections))]
    res_too_short = [[] for _ in range(len(cl.connections))]
    res_too_long_depth = [[] for _ in range(len(cl.connections))]
    res_count_invalid = np.zeros([len(cl.connections)], dtype=int)
    res_count_total = np.zeros([len(cl.connections)], dtype=int)

    for t, frame in enumerate(frames):
        for c, (i, j) in enumerate(cl.connections):
            if is_valid(frame[i]) and is_valid(frame[j]):
                res_count_total[c] += 1

                ds = np.linalg.norm(frame[i] - frame[j])
                dd = abs(frame[i][2] - frame[j][2])

                ds_min, ds_max = cl.lengths[(i, j)]
                dd_max = cl.depth_deviations[(i, j)]

                b = False
                if ds > ds_max:
                    res_too_long[c].append(ds - ds_max)
                    b = True
                if ds < ds_min:
                    res_too_short[c].append(ds_min - ds)
                    b = True
                if 0 < dd_max < dd:
                    res_too_long_depth[c].append(dd - dd_max)
                    b = True
                if b:
                    res_count_invalid[c] += 1

    res_invalid_share = res_count_invalid / res_count_total

    return res_too_long, res_too_short, res_too_long_depth, [[i] for i in res_invalid_share], [[i] for i in res_count_invalid]


def calculate_statistics(data: list[float], z_score_threshold=3):
    res = dict.fromkeys(["Data Points", "Mean", "Standart Deviation", "Median", "Min", "Max", "Deciles"])

    if len(data) > 1 and z_score_threshold > 0:
        z_scores = zscore(data)
        data = data[abs(z_scores) < z_score_threshold]

    res["Data Points"] = len(data)
    if len(data) > 0:
        res["Mean"] = statistics.mean(data)
        res["Median"] = statistics.median(data)
        res["Min"] = min(data)
        res["Max"] = max(data)
    if len(data) > 1:
        res["Standart Deviation"] = statistics.stdev(data)
        res["Deciles"] = statistics.quantiles(data=data, n=10)
    else:
        res["Standart Deviation"] = 0
        res["Deciles"] = [res["Mean"]] * 9

    return res


def save(data: list[list[float]], column_names: list[str], save_path: str):
    assert len(data) <= len(column_names), "Not enough column names"
    assert not os.path.exists(save_path)

    data = copy.deepcopy(data)
    max_len = max([len(d) for d in data])

    df = pd.DataFrame({})
    for i, d in enumerate(data):
        d.extend([None] * (max_len - len(d)))
        df[column_names[i]] = d

    if save_path.endswith(".csv"):
        df.to_csv(save_path, index=False)
    elif save_path.endswith(".xlsx"):
        df.to_excel(save_path, index=False)
    else:
        raise ValueError("save_path needs to end with .csv or .xlsx")

    print(f"{save_path} saved.")


def read_data_from_file(data_path):
    if data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    elif data_path.endswith(".xlsx"):
        data = pd.read_excel(data_path)
    else:
        raise ValueError("Provide csv or xlsx file.")
    return data


def read_data_from_files(data_paths):
    data = []
    for d in data_paths:
        data.append(read_data_from_file(d))
    return pd.concat(data, axis=1)


def print_statistics(data_path: str | list[str], z_score_threshold=3):
    if isinstance(data_path, str):
        data = read_data_from_file(data_path)
    elif isinstance(data_path, list):
        data = read_data_from_files(data_path)

    for dk in data.keys():
        print(f"\n{dk}:")
        stats = calculate_statistics(data[dk].dropna(), z_score_threshold)
        for k, v in stats.items():
            print(f"     {k}: {v}")
        print()


def show_statics(data_path: str | list[str], title="", x_axis_name="", y_axis_name="", z_score_threshold=3):
    if isinstance(data_path, str):
        data = read_data_from_file(data_path)
    elif isinstance(data_path, list):
        data = read_data_from_files(data_path)

    stats = {k: [] for k in ["Names", "Means", "StdDevs", "Medians", "DataPoints"]}

    for dk in data.keys():
        stats_raw = calculate_statistics(data[dk].dropna(), z_score_threshold)
        if stats_raw["Data Points"] > 0:
            stats["Names"].append(dk)
            stats["Means"].append(stats_raw["Mean"])
            stats["StdDevs"].append(stats_raw["Standart Deviation"])
            stats["Medians"].append(stats_raw["Median"])
            stats["DataPoints"].append(stats_raw["Data Points"])

    print(stats)
    df = pd.DataFrame(stats)

    # Creating a larger plot with two y-axes to accommodate more data and longer names
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed

    # Plotting means and medians on the first y-axis
    ax1.errorbar(df["Names"], df["Means"], yerr=df["StdDevs"], fmt='o', label='Mean ± Std Dev', color='blue')
    ax1.scatter(df["Names"], df["Medians"], color='red', label='Median')
    ax1.set_xlabel(x_axis_name)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_ylabel(y_axis_name, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(df["Names"], rotation=45, ha='right')  # Rotate names
    ax1.legend(loc='upper left')

    # Creating a second y-axis for the count of data points
    ax2 = ax1.twinx()
    ax2.bar(df["Names"], df["DataPoints"], color='green', alpha=0.3, label='Number of Data Points')
    ax2.set_ylabel('Count of Data Points', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')

    # Setting the title
    plt.title(title)

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    # Display the plot
    plt.show()


def create_dual_histogram(data, title="", x_label="", y_label="", y2_label="", start=None, end=None, interval=10):
    """
    Create a dual histogram with two y-axes:
    1. Normal histogram with blue points (left y-axis).
    2. Custom histogram with light green bars (right y-axis).

    Bins for data below and above the range are separated.
    """
    # Create bins
    if start is None:
        start = min(data) // 1
    if end is None:
        end = max(data) // 1 + 1

    # Create bins
    bins = np.arange(start, end + interval, interval)

    # Calculate normal histogram
    normal_counts, _ = np.histogram(data, bins)

    # Calculate custom histogram
    custom_heights = []
    for i in range(len(bins) - 1):
        bin_data = [d for d in data if bins[i] <= d < bins[i + 1]]
        if bin_data:
            custom_heights.append(sum(bin_data))
        else:
            custom_heights.append(0)

    # Create plot with two y-axes
    fig, ax1 = plt.subplots()

    # Normal histogram (blue points)
    ax1.plot(bins[:-1] + interval / 2, normal_counts, 'bo', label='Normal Histogram')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')

    # Custom histogram (light green bars)
    ax2 = ax1.twinx()
    ax2.bar(bins[:-1], custom_heights, width=interval, align='edge', color='lightgreen', edgecolor='black', label='Custom Histogram', alpha=0.5)
    ax2.set_ylabel(y2_label, color='lightgreen')
    ax2.tick_params(axis='y', labelcolor='lightgreen')

    plt.title(title)
    plt.show()


def create_histograms_for_all(data_path):
    data = read_data_from_file(data_path)
    for dk in data.keys():
        create_dual_histogram(data[dk], f"Histogram for {dk}-Joint", "Streak Time in s", "Amount of Streaks",
                              "Accumulated Streak Time", interval=5)
