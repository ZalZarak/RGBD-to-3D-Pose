import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
import statistics

import pandas as pd

import helper
import main

color_ranges = {
    "upper blue": 146,
    "upper green": 255,
    "upper red": 100,
    "lower blue": 0,
    "lower green": 150,
    "lower red": 0,
}

# Define the font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_thickness = 1
font_color = (255, 255, 255)  # White color

def debug_color_mask(resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 0, playback_file: str = None):
    def update_slider(var_name):
        new_val = int(sliders_gui[var_name].get())
        values_gui[var_name].set(new_val)
        color_ranges[var_name] = new_val

    def update_value(var_name, new_value):
        try:
            new_val = int(new_value)
            values_gui[var_name].set(new_val)
            sliders_gui[var_name].set(new_val)
            color_ranges[var_name] = new_val
        except ValueError:
            pass

    root = tk.Tk()
    root.title("Variable Manipulation")

    values_gui = {}
    sliders_gui = {}

    for var_name in color_ranges.keys():
        values_gui[var_name] = tk.IntVar(value=color_ranges[var_name])
        frame = tk.Frame(root)
        frame.pack()

        label = tk.Label(frame, text=var_name)
        label.pack(side=tk.LEFT)

        sliders_gui[var_name] = tk.Scale(frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=values_gui[var_name])
        sliders_gui[var_name].pack(side=tk.LEFT)

        entry = tk.Entry(frame, width=5)
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", lambda event, var_name=var_name, entry=entry: update_value(var_name, entry.get()))
        entry.bind("<FocusOut>", lambda event, var_name=var_name, entry=entry: update_value(var_name, entry.get()))

        sliders_gui[var_name].bind("<Motion>", lambda event, var_name=var_name: update_slider(var_name))

    cl = main.RGBDto3DPose(playback=playback_file is not None, duration=-1, playback_file=playback_file,
                           resolution=resolution, fps=fps, flip=flip, countdown=0, translation=None,
                           savefile_prefix=None, save_joints=False, save_bag=False,
                           show_rgb=False, show_depth=False, show_joints=False, show_color_mask=False,
                           simulate_limbs=False, simulate_joints=False, simulate_joint_connections=False)

    cl.prepare()

    # Initialize the image window
    cv2.namedWindow('Debug-Mask')
    cv2.namedWindow('Debug-Color')

    key = None
    try:
        while key != 27:
            color_frame, color_image, depth_frame, depth_image = cl.get_frames()

            helper.show_mask("Debug-Mask", color_image,
                             np.array([[color_ranges["lower blue"], color_ranges["lower green"], color_ranges["lower red"]],
                                       [color_ranges["upper blue"], color_ranges["upper green"], color_ranges["upper red"]]]))
            cv2.imshow("Debug-Color", color_image)

            root.update()

            key = cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        cl.pipeline.stop()


def debug_search_area(resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 0, playback_file: str = None):
    # mode 0: length, mode 1: depth, mode 2: search area

    def update_slider(var_name, value_idx):
        new_val = int(sliders_gui[var_name][value_idx].get())
        values_gui[var_name][value_idx].set(new_val)

        search_areas_hr[var_name][value_idx] = new_val
        search_areas[main.joint_map[var_name]] = helper.generate_base_search_area(search_areas_hr[var_name][0], search_areas_hr[var_name][1])

    def update_value(var_name, value_idx, new_value):
        try:
            new_val = int(new_value)
            values_gui[var_name][value_idx].set(new_val)
            sliders_gui[var_name][value_idx].set(new_val)

            search_areas_hr[var_name][value_idx] = new_val
            search_areas[main.joint_map[var_name]] = helper.generate_base_search_area(search_areas_hr[var_name][0], search_areas_hr[var_name][1])
        except ValueError:
            pass

    values_gui = {}
    sliders_gui = {}
    entry_fields = {}

    search_areas_hr = {k: list(v) for k, v in main.search_areas_hr.items()}
    search_areas = {k: list(v) for k, v in main.search_areas.items()}

    gui = tk.Tk()
    gui.title("Search Areas")

    canvas = tk.Canvas(gui)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(gui, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    for var_name, value_range in search_areas_hr.items():
        values_gui[var_name] = [tk.IntVar(value=value_range[0]), tk.IntVar(value=value_range[1])]  # Create two IntVar instances for each variable
        inner_frame = tk.Frame(frame)
        inner_frame.pack()

        label = tk.Label(inner_frame, text=var_name)
        label.pack(side=tk.LEFT)

        sliders_gui[var_name] = [tk.Scale(inner_frame, from_=0, to=40, orient=tk.HORIZONTAL, variable=values_gui[var_name][0]),
                                 tk.Scale(inner_frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=values_gui[var_name][1])]

        sliders_gui[var_name][0].pack(side=tk.LEFT)
        sliders_gui[var_name][1].pack(side=tk.LEFT)

        entry_frame = tk.Frame(inner_frame)
        entry_frame.pack(side=tk.LEFT)

        for value_idx in range(2):
            entry = tk.Entry(entry_frame, width=5)
            entry.pack(side=tk.LEFT)
            entry.bind("<Return>",
                       lambda event, var_name=var_name, value_idx=value_idx, entry=entry: update_value(var_name, value_idx, entry.get()))
            entry.bind("<FocusOut>",
                       lambda event, var_name=var_name, value_idx=value_idx, entry=entry: update_value(var_name, value_idx, entry.get()))

            entry_fields[(var_name, value_idx)] = entry

        # Bind slider updates to the slider movement
        sliders_gui[var_name][0].bind("<Motion>", lambda event, var_name=var_name, value_idx=0: update_slider(var_name, value_idx))
        sliders_gui[var_name][1].bind("<Motion>", lambda event, var_name=var_name, value_idx=1: update_slider(var_name, value_idx))


    cl = main.RGBDto3DPose(playback=playback_file is not None, duration=-1, playback_file=playback_file,
                           resolution=resolution, fps=fps, flip=flip, countdown=0, translation=None,
                           savefile_prefix=None, save_joints=False, save_bag=False,
                           show_rgb=False, show_depth=False, show_joints=True,
                           simulate_limbs=False, simulate_joints=False, simulate_joint_connections=False)

    cl.prepare()

    cv2.namedWindow("Debug-Search-Area")

    key = None
    try:
        while key != 27:
            color_frame, color_image, depth_frame, depth_image = cl.get_frames()
            joints_2d, confidences, joint_image = cl.openpose_handler.push_frame(color_image)
            joints_2d = joints_2d.astype(int)

            for i, joint2d in enumerate(joints_2d):
                color_image = color_image.copy()  # copy because cv2 cannot handle views of ndarrays e.g. if they were rotated etc.

                center_x, center_y = joint2d
                n = search_areas_hr[main.joint_map_rev[i]][0]*2 + 1
                top_left = (center_x - n // 2, center_y - n // 2)
                bottom_right = (center_x + n // 2, center_y + n // 2)
                cv2.rectangle(color_image, top_left, bottom_right, (64, 64, 64), 1)

                for x_search, y_search in helper.generate_search_pixels((center_x, center_y), i, search_areas, cl.resolution):
                    color_image[y_search, x_search] = (0, 0, 255)
                canvas.update_idletasks()
                gui.update()

            for pair in main.connections_list:
                point1, point2 = joints_2d[pair[0]], joints_2d[pair[1]]
                if point1.any() != 0 and point2.any() != 0:
                    cv2.line(color_image, point1, point2, (128, 128, 128), 2)

            cv2.imshow("Debug-Joint", color_image)

            key = cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        cl.pipeline.stop()


def debug_length(mode: int, output_filename: str = None, custom_connections=None, connections_except=None, resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 0,
                 playback_file: str = None):
    # mode 0: length, mode 1: depth

    assert mode in (0, 1), "mode is 0 or 1."
    assert connections_except is None or custom_connections is None, "Don't input connections_except and custom_connections at the same time"

    cl = main.RGBDto3DPose(playback=playback_file is not None, duration=-1, playback_file=playback_file,
                           resolution=resolution, fps=fps, flip=flip, countdown=0, translation=None,
                           savefile_prefix=None, save_joints=False, save_bag=False,
                           show_rgb=False, show_depth=False, show_joints=True,
                           simulate_limbs=False, simulate_joints=False, simulate_joint_connections=False)

    cl.prepare()

    cv2.namedWindow("Debug-Length")

    lengths = {}
    if mode == 0:
        lengths = {k: [] for k in main.lengths_hr.keys()}
    elif mode == 1:
        lengths = {k: [] for k in main.depth_deviations_hr.keys()}

    connections = main.connections_list
    if custom_connections is not None:
        connections = custom_connections
    elif connections_except is not None:
        connections = [c for c in connections if c not in connections_except]

    key = None
    try:
        while key != 27:
            color_frame, color_image, depth_frame, depth_image = cl.get_frames()
            joints_2d, confidences, joint_image = cl.openpose_handler.push_frame(color_image)
            joints_2d = joints_2d.astype(int)

            joints_3d = cl.get_3d_coords(joints_2d, depth_frame)

            for i, (joint2d, joint3d) in enumerate(zip(joints_2d, joints_3d)):
                if joint3d[2] != 0:
                    color_image = color_image.copy()  # copy because cv2 cannot handle views of ndarrays e.g. if they were rotated etc.
                    cv2.circle(color_image, joint2d, 3, (255, 255, 255), -1)
            for pair in connections:
                point1, point2 = joints_2d[pair[0]], joints_2d[pair[1]]
                point1_3d, point2_3d = joints_3d[pair[0]], joints_3d[pair[1]]
                lengths_key = main.joint_map_rev[pair[0]] + "-" + main.joint_map_rev[pair[1]]

                if point1.any() != 0 and point2.any() != 0 and point1_3d[2] != 0 and point2_3d[2] != 0:
                    center_x = (point1[0] + point2[0]) // 2
                    center_y = (point1[1] + point2[1]) // 2
                    if mode == 0:
                        l = np.linalg.norm(point1_3d - point2_3d)
                    else:
                        l = abs(point1_3d[2] - point2_3d[2])
                    color = (128, 128, 128) if main.lengths_hr[lengths_key][0] <= l <= main.lengths_hr[lengths_key][1] else (0, 0, 200)
                    lengths[lengths_key].append(l)
                    text = str(l.round(2))

                    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x = center_x - (text_size[0] // 2)
                    text_y = center_y + (text_size[1] // 2)
                    cv2.line(color_image, point1, point2, color, 2)
                    cv2.putText(color_image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

            cv2.imshow("Debug-Joint", color_image)

            key = cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        cl.pipeline.stop()

        stats = {}
        for name, l in lengths.items():
            if len(l) > 0:
                l.sort()
                mean = round(statistics.mean(l), 3)
                median = round(statistics.median(l), 3)
                min_ = round(l[0], 3)
                max_ = round(l[-1], 3)
                if len(l) > 1:
                    variance = round(statistics.variance(l), 3)
                    stddev = round(statistics.stdev(l), 3)
                    deciles = [round(i, 3) for i in statistics.quantiles(data=l, n=10)]
                else:
                    variance = ""
                    stddev = ""
                    deciles = [""] * 9

                print(f"""
                Connection: {name}
                    Data Points:        {len(l)}
                    Mean:               {mean}
                    Variance:           {variance}
                    Standard Deviation: {stddev}
                    Median:             {median}
                    Min:                {min_}
                    Max:                {max_}
                    Deciles:            {deciles}
                    """)

                stats[name] = {
                    "Data Points": len(l),
                    "Mean": mean,
                    "Variance": variance,
                    "Standard Deviation": stddev,
                    "Median": median,
                    "Min": min_,
                    "Max": max_,
                    "Deciles": deciles,
                }
            else:
                print(f"""
                Connection: {name}: No Data
                    """)

                stats[name] = {
                    "Data Points": 0,
                    "Mean": "",
                    "Variance": "",
                    "Standard Deviation": "",
                    "Median": "",
                    "Min": "",
                    "Max": "",
                    "Deciles": [""] * 9,
                }

            for i, decile in enumerate(stats[name]["Deciles"]):
                stats[name][f"Decile_{i}"] = decile
            stats[name].pop("Deciles")

        if output_filename is not None:
            max_length = max(len(col) if col is not None else 0 for col in lengths.values())
            for key, value in lengths.items():
                if len(value) < max_length:
                    lengths[key] += [""] * (max_length - len(value))
            df = pd.DataFrame(lengths)
            stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=list(stats[list(stats.keys())[0]].keys()))
            empty_columns = pd.Series([None] * len(df), name='')
            stats_df = pd.concat([stats_df, empty_columns], axis=1)
            stats_df = stats_df.transpose()
            combined_df = pd.concat([stats_df, df], axis=0)
            if output_filename.endswith(".csv"):
                combined_df.to_csv(output_filename, index=True)
            elif output_filename.endswith(".xlsx"):
                combined_df.to_excel(output_filename, index=True)

            print(f"Saved to {output_filename}.")


if __name__ == '__main__':
    debug_color_mask()
