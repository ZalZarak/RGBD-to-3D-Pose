import copy
import math
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from cfonts import render as render_text
from math import sin, cos, sqrt

from numpy import pi

import helper
from old.main import visualize_points
from openpose_handler import OpenPoseHandler
from simulator import simulate_sync

import multiprocessing as mp

"""lengths = {
    "neck-nose": 0.2,
    "neck-eye": 0.26,
    "neck-ear": 0.23,
    "nose-eye": 0.05,
    "eye-ear": 0.1,
    "neck-shoulder": 0.19,
    "shoulder-elbow": 0.3,
    "elbow-wrist": 0.29,
    "neck-midhip": 0.52,
    "midhip-hip": 0.11,
    "neck-hip": sqrt(0.52**2 + 0.13**2),
    "hip-knee": 0.42,
    "knee-foot": 0.46,
    "foot-heel": 0.07,
    "foot-toe": 0.2
}

lengths = {
    "neck-nose": (0.2, 0.075, 0.075),
    "nose-eye": (0.05, 0.02, 0.01),
    "eye-ear": (0.1, 0.03, 0.02),
    "neck-shoulder": (0.19, 0.04, 0.02),
    "shoulder-elbow": (0.3, 0.03, 0.06),
    "elbow-wrist": (0.29, 0.03, 0.03),
    "neck-midhip": (0.52, 0.15, 0.075),
    "midhip-hip": (0.11, 0.03, 0.02),
    "hip-knee": (0.42, 0.04, 0.07),
    "knee-foot": (0.46, 0.02, 0.05),
    "foot-toe": (0.2, 0.05, 0.01)
}"""

# as defined in OpenPose
joint_map = {
    'Nose': 0,
    'Neck': 1,
    'RShoulder': 2,
    'RElbow': 3,
    'RWrist': 4,
    'LShoulder': 5,
    'LElbow': 6,
    'LWrist': 7,
    'MidHip': 8,
    'RHip': 9,
    'RKnee': 10,
    'RAnkle': 11,
    'LHip': 12,
    'LKnee': 13,
    'LAnkle': 14,
    'REye': 15,
    'LEye': 16,
    'REar': 17,
    'LEar': 18,
    'LBigToe': 19,
    'LSmallToe': 20,
    'LHeel': 21,
    'RBigToe': 22,
    'RSmallToe': 23,
    'RHeel': 24,
    'Background': 25}

joint_map_rev = {v: k for k, v in joint_map.items()}

# must be: for all 'j1-j2': joint_map[j1] < joint_map[j2]
lengths_hr = {
    'Nose-Neck': (0.125, 0.275),
    'Nose-LEye': (0.03, 0.06),
    'Nose-REye': (0.03, 0.06),
    'LEye-LEar': (0.07, 0.12),  # with this deviation down it will invalidate front view of face but make it robuster to occlusion. If front, eye detection should work
    'REye-REar': (0.07, 0.12),
    'Neck-LShoulder': (0.15, 0.21),
    'Neck-RShoulder': (0.15, 0.21),
    'LShoulder-LElbow': (0.27, 0.36),
    'RShoulder-RElbow': (0.27, 0.36),
    'LElbow-LWrist': (0.26, 0.32),
    'RElbow-RWrist': (0.26, 0.32),
    'Neck-MidHip': (0.37, 0.595),
    'MidHip-LHip': (0.1, 0.13),
    'MidHip-RHip': (0.1, 0.13),
    'LHip-LKnee': (0.38, 0.49),
    'RHip-RKnee': (0.38, 0.49),
    'LKnee-LAnkle': (0.44, 0.51),
    'RKnee-RAnkle': (0.44, 0.51),
    'LAnkle-LBigToe': (0.15, 0.21),
    'RAnkle-RBigToe': (0.15, 0.21)
}

# must be: for all 'j1-j2': joint_map[j1] < joint_map[j2]
depth_deviations_hr = {
    'Nose-Neck': 0.2,
    'Nose-LEye': 0.04,
    'Nose-REye': 0.04,
    'LEye-LEar': 0.07,  # with this deviation down it will invalidate front view of face but make it robuster to occlusion. If front, eye detection should work
    'REye-REar': 0.07,
    'Neck-LShoulder': 0.17,
    'Neck-RShoulder': 0.17,
    'LShoulder-LElbow': -1,
    'RShoulder-RElbow': -1,
    'LElbow-LWrist': -1,
    'RElbow-RWrist': -1,
    'Neck-MidHip': 0.2,
    'MidHip-LHip': 0.09,
    'MidHip-RHip': 0.09,
    'LHip-LKnee': -1,
    'RHip-RKnee': -1,
    'LKnee-LAnkle': -1,
    'RKnee-RAnkle': -1,
    'LAnkle-LBigToe': -1,
    'RAnkle-RBigToe': -1
}

# must be: for all 'j1-j2': joint_map[j1] < joint_map[j2]
search_areas_hr = {     # (deviation, skip) in pixels
    'Nose': (8, 3),
    'Neck': (12, 3),
    'RShoulder': (9, 2),
    'RElbow': (12, 2),
    'RWrist': (12, 2),
    'LShoulder': (9, 2),
    'LElbow': (12, 2),
    'LWrist': (12, 2),
    'MidHip': (15, 4),
    'RHip': (6, 5),
    'RKnee': (5, 4),
    'RAnkle': (4, 3),
    'LHip': (6, 5),
    'LKnee': (5, 4),
    'LAnkle': (4, 3),
    'REye': (0, 0),
    'LEye': (0, 0),
    'REar': (0, 0),
    'LEar': (0, 0),
    'LBigToe': (0, 0),
    'LSmallToe': (0, 0),
    'LHeel': (0, 0),
    'RBigToe': (0, 0),
    'RSmallToe': (0, 0),
    'RHeel': (0, 0),
    'Background': (0, 0)
}

lengths = {}
for k, v in lengths_hr.items():
    k1, k2 = k.split('-')
    lengths[(joint_map[k1], joint_map[k2])] = v

depth_deviations = {}
for k, v in depth_deviations_hr.items():
    k1, k2 = k.split('-')
    depth_deviations[(joint_map[k1], joint_map[k2])] = v

search_areas = {}
for k, v in search_areas_hr.items():
    search_areas[joint_map[k]] = helper.generate_base_search_area(v[0], v[1])

# for all tuples (a,b): a<b
connections_list = [(0, 1), (0, 15), (0, 16), (15, 17), (16, 18), (1, 2), (1, 5), (2, 3), (5, 6), (3, 4), (6, 7),
                    (1, 8), (8, 9), (8, 12), (9, 10), (12, 13), (10, 11), (13, 14), (11, 22), (14, 19)]

connections_dict = {k: [] for k in range(26)}
for a, b in connections_list:
    connections_dict[a].append(b)
    connections_dict[b].append(a)

color_range = np.array([[0, 135, 0], [255, 255, 75]])

color_validation_joints_hr = ["RWrist", "LWrist"]
color_validation_joints = [joint_map[j] for j in color_validation_joints_hr]


class RGBDto3DPose:
    count = 0

    def __init__(self, playback: bool, duration: float, playback_file: str | None,
                 resolution: tuple[int, int], fps: int, flip: int, countdown: int,
                 translation: (float, float, float), rotation: (float, float, float),
                 savefile_prefix: str | None, save_joints: bool, save_bag: bool,
                 show_rgb: bool, show_depth: bool, show_joints: bool, show_color_mask: bool,
                 simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool):
        self.playback = playback
        self.duration = duration
        self.playback_file = playback_file

        translation = np.array(translation)
        # self.transform = lambda m: m + translation if m[2] != 0 else m

        rotx, roty, rotz = rotation #tuple(map(lambda x: -x, rotation))
        x_rot_mat = np.array([
            [1,         0,          0],
            [0, cos(rotx), -sin(rotx)],
            [0, sin(rotx),  cos(rotx)]
        ])
        y_rot_mat = np.array([
            [cos(roty),  0, sin(roty)],
            [0,          1,         0],
            [-sin(roty), 0, cos(roty)]
        ])
        z_rot_mat = np.array([
            [cos(rotz), -sin(rotz), 0],
            [sin(rotz),  cos(rotz), 0],
            [0,          0,         1]
        ])
        rot_mat = x_rot_mat @ y_rot_mat @ z_rot_mat

        if rotation == (0, 0, 0) and translation == (0, 0, 0):  # less calculation
            self.transform = lambda m: m
        elif rotation == (0, 0, 0):                             # less calculation
            self.transform = lambda m: m + translation if m[2] != 0 else m
        else:
            self.transform = lambda m: rot_mat @ (m + translation) if m[2] != 0 else m

        self.resolution = resolution
        self.fps = fps
        self.flip = flip
        self.countdown = countdown
        self.save_joints = save_joints
        self.save_bag = save_bag
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.show_joints = show_joints
        self.show_color_mask = show_color_mask

        self.use_openpose = save_joints or show_joints
        self.colorizer = rs.colorizer()  # Create colorizer object

        if playback and (duration is None or duration <= 0):
            self.duration = float("inf")

        # define (inverse) rotation function to call deprojection at the correct pixel
        if flip in [-4, 0, 4]:  # no rotation
            self.inverse_flip = lambda x, y: (round(x), round(y))
            self.flip_3d_coord = lambda c: (c[0], -c[1], c[2])
            self.flip_rev = 0
        elif flip in [-3, 1]:  # left rotation
            self.inverse_flip = lambda x, y: (resolution[0] - round(y), round(x))
            self.flip_3d_coord = lambda c: (c[1], c[0], c[2])
            self.flip_rev = -1
        elif flip in [-2, 2]:  # 180° rotation
            self.inverse_flip = lambda x, y: (resolution[0] - round(x), resolution[1] - round(y))
            self.flip_3d_coord = lambda c: (-c[0], c[1], c[2])
            self.flip_rev = 2
        elif flip in [-1, 3]:  # right rotation
            self.inverse_flip = lambda x, y: (round(y), resolution[1] - round(x))
            self.flip_3d_coord = lambda c: (-c[1], -c[0], c[2])
            self.flip_rev = 1
        else:
            raise ValueError("Rotation should be in range [-4, 4]")

        if (save_bag or save_joints) and (savefile_prefix is None or savefile_prefix == ""):
            raise ValueError("Provide prefix for saving files")
        if playback and (playback_file is None or playback_file == ""):
            raise ValueError("Provide playback file")

        self.filename_bag = savefile_prefix + ".bag" if save_bag else ""
        self.filename_joints = savefile_prefix + "_joints.pkl" if save_joints else ""

        if save_bag and os.path.exists(self.filename_bag):
            print(f"File {self.filename_bag} already exists. Proceeding will overwrite this file.")
            print(f"Proceed? y/[n]")
            if input().lower() != 'y':
                exit()

        if save_joints and os.path.exists(self.filename_joints):
            print(f"File {self.filename_bag} already exists. Proceeding will overwrite this file.")
            print(f"Proceed? y/[n]")
            if input().lower() != 'y':
                exit()

        if show_rgb:
            cv2.namedWindow("RGB-Stream", cv2.WINDOW_AUTOSIZE)
        if show_depth:
            cv2.namedWindow("Depth-Stream", cv2.WINDOW_AUTOSIZE)

        self.intrinsics = None
        self.pipeline = None
        self.openpose_handler: OpenPoseHandler = None

        self.simulate = simulate_limbs or simulate_joints or simulate_joint_connections
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections
        self.done = mp.Value('b', False)
        self.joints = None

        self.joints_save = []
        self.start_time = -1

    def run(self):
        print("Use ESC to terminate, otherwise no files will be saved.")
        try:
            max_frames = int(self.duration * self.fps)  # frames = seconds * fps
        except OverflowError:
            max_frames = self.duration
        frame_counter = 0
        key = None

        self.prepare()
        self.start_time = time.time()

        try:
            while key != 27 and frame_counter < max_frames and not self.done.value:
                self.process_frame()

                key = cv2.waitKey(1)
        finally:
            self.done.value = True
            cv2.destroyAllWindows()
            self.pipeline.stop()
            if self.save_joints:
                with open(self.filename_joints, 'wb') as file:
                    pickle.dump(self.joints_save, file)
                print("Joints saved to:", self.filename_joints)


    def prepare(self):
        # Prepare visualizer

        if self.simulate:
            """visualizer_conn, self.main_conn = mp.Pipe(duplex=False)
            ready = mp.Event()
            self.p_vis = mp.Process(target=visualize, args=(visualizer_conn, ready, self.simulate_shape, self.simulate_joints, self.simulate_joint_connections))
            self.p_vis.start()
            ready.wait()"""

            ready = mp.Event()
            self.joints = mp.Array('f', np.zeros([25 * 3]))
            simulator_process = mp.Process(target=simulate_sync, args=(
            self.joints, ready, self.done, self.simulate_limbs, self.simulate_joints, self.simulate_joint_connections))
            simulator_process.start()
            ready.wait()

        # Initialize OpenPose Handler
        if self.use_openpose:
            self.openpose_handler = OpenPoseHandler()

        # Initialize the RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()

        if self.playback:
            # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
            rs.config.enable_device_from_file(config, file_name=self.playback_file)

        # Enable both depth and color streams
        config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)
        self.align = rs.align(rs.stream.color)
        self.decimation_filter = rs.decimation_filter()
        self.depth2disparity = rs.disparity_transform(True)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.disparity2depth = rs.disparity_transform(False)
        self.hole_filling_filter = rs.hole_filling_filter(2)  # i feel like this is buggy

        if self.save_bag:
            config.enable_record_to_file(self.filename_bag)
            # Start the pipeline
            pipeline_profile = pipeline.start(config)
            device = pipeline_profile.get_device()
            recorder = device.as_recorder()
            rs.recorder.pause(recorder)
            helper.print_countdown(self.countdown)
            rs.recorder.resume(recorder)
        else:
            if self.countdown > 0:
                helper.print_countdown(self.countdown)
            pipeline_profile = pipeline.start(config)

        self.intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        if not self.playback:
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            for i in range(int(depth_sensor.get_option_range(rs.option.visual_preset).max)):
                if depth_sensor.get_option_value_description(rs.option.visual_preset, i) == "High Density":
                    depth_sensor.set_option(rs.option.visual_preset, i)
            depth_sensor.set_option(rs.option.exposure, 16000)
            depth_sensor.set_option(rs.option.gain, 16)
            depth_sensor.set_option(rs.option.laser_power, 360)
            depth_sensor.set_option(rs.option.depth_units, 0.0005)

        self.pipeline = pipeline

    def process_frame(self):
        color_frame, color_image, depth_frame, depth_image = self.get_frames()

        if self.use_openpose:
            # get joints from OpenPose. Assume there is only one person
            joints_2d, confidences, joint_image = self.openpose_handler.push_frame(color_image)

            if self.show_rgb:
                helper.show("RGB-Stream", joint_image if self.show_joints else color_image)
            if self.show_depth:
                helper.show("Depth-Stream", depth_image, joints_2d if self.show_joints else None)

            joints_3d = self.get_3d_coords(joints_2d, depth_frame)
            joints_val = self.validate_joints(joints_3d, joints_2d, confidences, depth_frame, color_image)
            joints_val = np.apply_along_axis(self.flip_3d_coord, 1, joints_val)     # flip the coordinates into the right perspective
            joints_val = np.apply_along_axis(self.transform, 1, joints_val)         # apply translation/rotation

            if self.simulate:
                self.joints[:] = joints_val.flatten()
            if self.save_joints:
                self.joints_save.append((time.time() - self.start_time, joints_val))

            """if self.count == 10:
                # visualize_points(joints_val, OpenPoseHandler.pairs, joints)
                visualize(joints, OpenPoseHandler.pairs)
            self.count += 1"""
        else:
            if self.show_rgb:
                helper.show("RGB-Stream", color_image)
            if self.show_depth:
                helper.show("Depth-Stream", depth_image)
        if self.show_color_mask:
            helper.show_mask("Color-Mask-Stream", color_image, color_range)

    def get_frames(self) -> tuple[any, np.ndarray, any, np.ndarray]:
        # Wait for the next set of frames from the camera
        # self.align = rs.align(rs.stream.color)
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        # depth_frame = self.decimation_filter.process(depth_frame)
        depth_frame = self.depth2disparity.process(depth_frame)
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.disparity2depth.process(depth_frame)
        # depth_frame = self.hole_filling_filter.process(depth_frame)
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = self.colorizer.colorize(depth_frame)

        depth_frame = depth_frame.as_depth_frame()

        # Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.rot90(depth_image, k=self.flip)
        color_image = np.rot90(color_image, k=self.flip)

        return color_frame, color_image, depth_frame, depth_image

    def get_3d_coords(self, joints: np.ndarray, depth_frame) -> np.ndarray:
        coords = np.zeros([joints.shape[0], 3])

        for i, (x, y) in enumerate(joints):
            try:
                x, y = self.inverse_flip(x, y)
                depth = depth_frame.get_distance(x, y)
                if depth > 0:
                    # get 3d coordinates and reorder them from y,x,z to x,y,z
                    coords[i] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)
                else:
                    coords[i] = 0
            except RuntimeError:  # joint outside of picture
                pass

        return coords

    def validate_joints(self, joints_3d: np.ndarray, joints_2d: np.ndarray, confidences: np.ndarray, depth_frame, color_image: np.ndarray) -> np.ndarray:
        def validate_joint(connection: tuple[int, int]) -> bool:
            if val_joints[connection[0], 2] == 0 or val_joints[connection[1], 2] == 0:
                return False

            l_min, l_max = lengths[connection]
            deviation = depth_deviations[connection]

            return ((l_min <= np.linalg.norm(val_joints[connection[1]] - val_joints[connection[0]]) <= l_max)  # length validation
                   and
                   (deviation < 0 or abs(val_joints[connection[0], 2] - val_joints[connection[1], 2]) <= deviation))  # depth validation

        val = np.zeros(25, dtype="bool")

        val_joints = np.copy(joints_3d)

        # If depth is wrong, then the length of limbs should be incorrect or the depth deviation is too high
        # The body should have correct length
        # for some connections also the same depth

        # validate each joint through at least one connection which is valid by length and depth
        for connection in connections_list:
            success = validate_joint(connection)
            # if validation is successful mark both joints as validated, otherwise leave them as is
            val[connection[0]] |= success
            val[connection[1]] |= success

        # search around the joint for the right color
        # if found, define the 3d coordinate of this joint by original x,y and the found depth
        # this will accept any positive depth without further validation
        # make sure to have very precise color range, e.g. detected pixels not belonging to corresponding limbs should be nearly zero
        color_image = np.rot90(color_image, k=self.flip_rev)    # rotate color_image back to fit the depth_frame
        for color_joint in color_validation_joints:     # iterate through corresponding, unvalidated joints
            if not val[color_joint]:
                # flip the current pixel to camera coordinate system
                x_flipped, y_flipped = self.inverse_flip(joints_2d[color_joint, 0], joints_2d[color_joint, 1])
                # generate search pixels around joint
                for x_search, y_search in helper.generate_search_pixels((x_flipped, y_flipped), color_joint, search_areas, self.resolution):
                    color = color_image[y_search, x_search]
                    if all(np.greater_equal(color_range[1], color)) and all(np.greater_equal(color, color_range[0])):
                        try:
                            depth = depth_frame.get_distance(x_search, y_search)    # get the depth at this pixel
                            if depth > 0:
                                # get coordinate of original x,y but with the new depth
                                val_joints[color_joint] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x_flipped, y_flipped), depth=depth)
                                val[color_joint] = True
                                break   # found depth, go to next joint
                        except RuntimeError:  # joint outside of image
                            print("This shouldn't happen during color correction!")

        # search around the joint for the right depth e.g. where it validates through one connection
        # if found, define the 3d coordinate of this joint by original x,y and the found depth
        # try correcting until no joint was corrected
        # theoretically it might run into infinity loop, but it always ran smoothly
        change = True
        while change:
            change = False
            for i in range(25):
                # if joint is detected but not validated try to correct depth
                if not val[i]:
                    # flip the current pixel to camera coordinate system
                    x_flipped, y_flipped = self.inverse_flip(joints_2d[i, 0], joints_2d[i, 1])
                    # generate search pixels around joint
                    for x_search, y_search in helper.generate_search_pixels((x_flipped, y_flipped), i, search_areas, self.resolution):    # for each pixel in search area
                        try:
                            depth = depth_frame.get_distance(x_search, y_search)    # get the depth at this pixel
                            if depth <= 0:
                                continue

                            # get coordinate of original x,y but with the new depth
                            val_joints[i] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x_flipped, y_flipped), depth=depth)

                            # try if any connection validates successfully then break out and continue with the next joint
                            for connected_joint in connections_dict[i]:
                                j1, j2 = sorted((i, connected_joint))
                                if validate_joint((j1, j2)):
                                    val[j1] = val[j2] = True  # to not correct those joints again
                                    change = True
                                    break  # Break the inner loop...
                            else:
                                continue  # Continue if the inner loop wasn't broken.
                            break  # Inner loop was broken, break the outer.
                        except RuntimeError:  # joint outside of image
                            print("This shouldn't happen during correction!")
                            pass

        for i in range(25):
            # set supposedly incorrect joints to zero
            if not val[i]:
                val_joints[i] = 0

        # reduce head to nose
        if all(val_joints[0] == 0):  # nose not detected
            if val_joints[15, 2] != 0 and val_joints[16, 2] != 0:  # both eyes validated
                val_joints[0] = (val_joints[15] + val_joints[16]) / 2
            elif val_joints[15, 2] != 0:  # one eye validated
                val_joints[0] = val_joints[15]
            elif val_joints[16, 2] != 0:  # one eye validated
                val_joints[0] = val_joints[16]
            elif val_joints[17, 2] != 0 and val_joints[18, 2] != 0:  # both ears validated
                val_joints[0] = (val_joints[17] + val_joints[18]) / 2
            elif val_joints[17, 2] != 0:  # one ear validated
                val_joints[0] = val_joints[17]
            elif val_joints[18, 2] != 0:  # one ear validated
                val_joints[0] = val_joints[18]
        elif val_joints[0, 2] == 0:  # nose not validated
            if val_joints[15, 2] != 0 and val_joints[16, 2] != 0:  # both eyes validated
                val_joints[0, 2] = (val_joints[15, 2] + val_joints[16, 2]) / 2
            elif val_joints[15, 2] != 0:  # one eye validated
                val_joints[0, 2] = val_joints[15, 2]
            elif val_joints[16, 2] != 0:  # one eye validated
                val_joints[0, 2] = val_joints[16, 2]
            elif val_joints[17, 2] != 0 and val_joints[18, 2] != 0:  # both ears validated
                val_joints[0, 2] = (val_joints[17, 2] + val_joints[18, 2]) / 2
            elif val_joints[17, 2] != 0:  # one ear validated
                val_joints[0, 2] = val_joints[17, 2]
            elif val_joints[18, 2] != 0:  # one ear validated
                val_joints[0, 2] = val_joints[18, 2]

        return val_joints


def stream(savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = float("inf"),
           resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 1, countdown: int = 0, translation=(0, 0, 0), rotation=(0, 0, 0),
           show_rgb: bool = True, show_depth: bool = True, show_joints: bool = True, show_color_mask: bool = False,
           simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=False, duration=duration, playback_file=None,
                      resolution=resolution, fps=fps, flip=flip, countdown=countdown, translation=translation, rotation=rotation,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joints=show_joints, show_color_mask=show_color_mask,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


def playback(playback_file: str, savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = -1,
             resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 1, translation=(0, 0, 0), rotation=(0, 0, 0),
             show_rgb: bool = True, show_depth: bool = True, show_joints: bool = True, show_color_mask: bool = False,
             simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=True, duration=duration, playback_file=playback_file,
                      resolution=resolution, fps=fps, flip=flip, countdown=0, translation=translation, rotation=rotation,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joints=show_joints, show_color_mask=show_color_mask,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


if __name__ == '__main__':
    cam_translation = (0, 1.5, 0)  # from (x,y,z) = (0,0,0) in m
    cam_rotation = (0, 0, 0)  # from pointing parallel to z-axis, (x-rotation [up/down], y-rotation [left/right], z-rotation/tilt [anti-clockwise, clockwise]), in radians

    # playback("test.bag", save_joints=True, savefile_prefix="vid", simulate_limbs = False, simulate_joints = False, simulate_joint_connections = False)
    # playback("test.bag", translation=cam_translation, simulate_joint_connections=False, simulate_joints=False)

    # stream(countdown=3, translation=cam_translation, rotate=2, show_rgb=True, show_depth=True, show_joints=True, simulate_joints=False, simulate_joint_connections=False, simulate_limbs=True)

    stream(translation=cam_translation, rotation=cam_rotation, flip=0, show_rgb=True, show_depth=True, show_joints=True, simulate_joints=False,
           simulate_joint_connections=False, simulate_limbs=True, show_color_mask=True)

    #stream(savefile_prefix="test_val", save_joints = False, save_bag = True, flip = 0, countdown = 2, translation=cam_translation,
     #      show_rgb = True, show_depth = True, show_joints = True,
      #     simulate_limbs = False, simulate_joints = False, simulate_joint_connections = False)

    # playback("test_val.bag", flip=0, translation=cam_translation, simulate_joint_connections=False, simulate_joints=False)