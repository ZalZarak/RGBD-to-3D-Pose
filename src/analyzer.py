import pickle
import numpy as np

from src.config import config
from src.perceptor import Perceptor

conf = config["Perceptor"]
conf["playback"] = True
conf["playback_file"] = " "
conf["countdown"] = 0
conf["save_joints"] = conf["save_bag"] = conf["show_rgb"] = conf["show_depth"] = conf["show_joints"] = conf["show_color_mask"] \
    = conf["simulate_limbs"] = conf["simulate_joints"] = conf["simulate_joint_connections"] = False

cl = Perceptor(**conf)

joint_file = "res/training_2_joints.pkl"
with open(joint_file, "rb") as f:
    obj = list(pickle.load(f))#[0:3]
times = np.array(list(map(lambda f: f[0], obj)))
frames = np.array(list(map(lambda f: f[1], obj)))
l = len(obj)

is_valid = lambda j: np.all(j != (0,0,0))

def count_valid_joints():


    """frames = map(lambda f: f[1], obj)
    res = np.zeros([25], dtype=int)
    for frame in frames:
        res += np.any(frame != [0,0,0], axis=1)"""

    res = np.sum(np.all(frames != [0, 0, 0], axis=-1), axis=0).astype(int)
    print(res)


def count_streak_times():
    res = [[] for _ in range(25)]
    on_streak = np.ones([25], dtype=float)*-1   # strike start time

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
    print(res)


def get_joint_speeds():
    res = [[] for _ in range(25)]

    for t, frame in enumerate(frames[1:]):
        t += 1
        for i, joint in enumerate(frame):
            if is_valid(joint) and is_valid(frames[t-1,i]):
                ds = np.linalg.norm(joint - frames[t-1,i])
                dt = times[t] - times[t-1]
                res[i].append(ds/dt)
    print(res)


def get_invalid_connections():
    res_too_long = [[] for _ in range(len(cl.connections))]
    res_too_short = [[] for _ in range(len(cl.connections))]
    res_too_long_depth = [[] for _ in range(len(cl.connections))]
    res_count = np.zeros([len(cl.connections)], dtype=int)
    res_test = np.zeros([len(cl.connections)], dtype=int)

    for t, frame in enumerate(frames):
        for c, (i, j) in enumerate(cl.connections):
            if is_valid(frame[i]) and is_valid(frame[j]):
                res_test[c] += 1

                ds = np.linalg.norm(frame[i] - frame[j])
                dd = abs(frame[i][2] - frame[j][2])

                ds_min, ds_max = cl.lengths[(i, j)]
                dd_max = cl.depth_deviations[(i, j)]

                b = False
                if ds > ds_max:
                    res_too_long[c].append(ds-ds_max)
                    b = True
                if ds < ds_min:
                    res_too_short[c].append(ds_min-ds)
                    b = True
                if 0 < dd_max < dd:
                    res_too_long_depth[c].append(dd-dd_max)
                    b = True
                if b:
                    res_count[c] += 1

    def p(data):
        for i, d in enumerate(data):
            print(f"{cl.connections_hr[i]}: {d}")

    print("TEST:")
    p(res_test)
    print("COUNT: ")
    p(res_count)
    input()
    print("\nTOO LONG:")
    p(map(lambda i: len(i), res_too_long))
    input()
    print("\nTOO SHORT:")
    p(map(lambda i: len(i), res_too_short))
    input()
    print("\nDEPTH TO LONG:")
    p(map(lambda i: len(i), res_too_long_depth))


if __name__ == '__main__':
    get_invalid_connections()
