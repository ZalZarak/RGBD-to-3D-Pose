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

import helper
import pybullet_simulation as sim
from helper import print_countdown, draw_pixel_grid
from old.main import visualize_points
from openpose_handler import OpenPoseHandler
from simulator import simulate_sync

import multiprocessing as mp


lengths = {
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

class RGBDto3DPose:
    count = 0

    def __init__(self, playback: bool, duration: float, playback_file: str | None,
                 resolution: tuple[int, int], fps: int, flip: int, countdown: int, translation: (float, float, float),
                 savefile_prefix: str | None, save_joints: bool, save_bag: bool,
                 show_rgb: bool, show_depth: bool, show_joints: bool,
                 simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool):
        self.playback = playback
        self.duration = duration
        self.playback_file = playback_file

        translation = np.array(translation)
        self.transform = lambda m: m + translation if m[2] != 0 else m

        """rotx, roty, rotz = rotation #tuple(map(lambda x: -x, rotation))
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

        if rotation == (0, 0, 0):
            self.transform = lambda m: m + translation if m[2] != 0 else m
        else:
            self.transform = lambda m: rot_mat @ (m + translation) if m[2] != 0 else m"""

        self.resolution = resolution
        self.fps = fps
        self.flip = flip
        self.countdown = countdown
        self.save_joints = save_joints
        self.save_bag = save_bag
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.show_joints = show_joints

        self.use_openpose = save_joints or show_joints
        self.colorizer = rs.colorizer()  # Create colorizer object

        if playback and (duration is None or duration <= 0):
            self.duration = float("inf")

        # define (inverse) rotation function to call deprojection at the correct pixel
        if flip in [-4, 0, 4]:  # no rotation
            self.inverse_flip = lambda x, y: (round(x), round(y))
            self.flip_3d_coord = lambda c: (c[0], -c[1], c[2])
        elif flip in [-3, 1]:  # left rotation
            self.inverse_flip = lambda x, y: (resolution[0] - round(y), round(x))
            self.flip_3d_coord = lambda c: (c[1], c[0], c[2])
        elif flip in [-2, 2]:  # 180° rotation
            self.inverse_flip = lambda x, y: (resolution[0] - round(x), resolution[1] - round(y))
            self.flip_3d_coord = lambda c: (-c[0], c[1], c[2])
        elif flip in [-1, 3]:  # right rotation
            self.inverse_flip = lambda x, y: (round(y), resolution[1] - round(x))
            self.flip_3d_coord = lambda c: (-c[1], -c[0], c[2])
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
            simulator_process = mp.Process(target=simulate_sync, args=(self.joints, ready, self.done, self.simulate_limbs, self.simulate_joints, self.simulate_joint_connections))
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
        self.hole_filling_filter = rs.hole_filling_filter(1)    # i feel like this is buggy

        if self.save_bag:
            config.enable_record_to_file(self.filename_bag)
            # Start the pipeline
            pipeline_profile = pipeline.start(config)
            device = pipeline_profile.get_device()
            recorder = device.as_recorder()
            rs.recorder.pause(recorder)
            print_countdown(self.countdown)
            rs.recorder.resume(recorder)
        else:
            if self.countdown > 0:
                print_countdown(self.countdown)
            pipeline_profile = pipeline.start(config)

        self.intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

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
        depth_frame = self.hole_filling_filter.process(depth_frame)
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = self.colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.rot90(depth_image, k=self.flip)
        color_image = np.rot90(color_image, k=self.flip)

        if self.use_openpose:
            # get joints from OpenPose. Assume there is only one person
            res, joint_image = self.openpose_handler.push_frame(color_image)
            joints, confidences = res[0, :, :2], res[0, :, 2]

            if self.show_rgb:
                helper.show("RGB-Stream", joint_image if self.show_joints else color_image)
            if self.show_depth:
                helper.show("Depth-Stream", depth_image, joints if self.show_joints else None)

            joints = self.get_3d_coords(joints, depth_frame)
            joints_val = self.validate_joints(joints, confidences)
            joints_val = np.apply_along_axis(self.transform, 1, joints_val)

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

    def get_3d_coords(self, joints: np.ndarray, depth_frame) -> np.ndarray:
        depths = depth_frame.as_depth_frame()
        coords = np.zeros([joints.shape[0], 3])

        for i, (x, y) in enumerate(joints):
            try:
                x, y = self.inverse_flip(x, y)
                depth = depths.get_distance(x, y)

                # get 3d coordinates and reorder them from y,x,z to x,y,z
                coord = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)
                coords[i] = self.flip_3d_coord(coord)  # coord[1], coord[0], coord[2]
            except RuntimeError:  # joint outside of picture
                pass

        return coords


    def validate_joints(self, joints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        def val_length(joint1: np.ndarray, joint2: np.ndarray, expected_length: float, deviation_down: float = 0.2, deviation_up: float = 0.2) -> bool:
            return expected_length - deviation_down <= np.linalg.norm(joint2 - joint1) <= expected_length + deviation_up

        def val_depth(joint1: np.ndarray, joint2: np.ndarray, deviation: float = 0.2) -> bool:
            return abs(joint1[2] - joint2[2]) <= deviation

        val = np.zeros((25, 25), dtype="bool")

        val_joints = np.copy(joints)

        # If depth is wrong, then the length of limbs should be incorrect or the depth deviation is too high

        # The upper body should have correct length and have approximately the same depth
        # neck-midhip
        val[1, 8] = val_length(joints[1], joints[8], lengths["neck-midhip"], 0.15, 0.075) and val_depth(joints[1], joints[8], 0.4)
        # neck-nose
        val[0, 1] = val_length(joints[0], joints[1], lengths["neck-nose"], 0.075, 0.075) and val_depth(joints[0], joints[1], 0.2)
        # neck-shoulder
        val[1, 2] = val_length(joints[1], joints[2], lengths["neck-shoulder"], 0.04, 0.02) and val_depth(joints[1], joints[2], 0.17)
        val[1, 5] = val_length(joints[1], joints[5], lengths["neck-shoulder"], 0.04, 0.02) and val_depth(joints[1], joints[5], 0.17)
        # midhip-hip
        val[8, 9] = val_length(joints[8], joints[9], lengths["midhip-hip"], 0.03, 0.02) and val_depth(joints[8], joints[9], 0.09)
        val[8, 12] = val_length(joints[8], joints[12], lengths["midhip-hip"], 0.03, 0.02) and val_depth(joints[8], joints[12], 0.09)

        # nose-eye
        val[0, 15] = val_length(joints[0], joints[15], lengths["nose-eye"], 0.02, 0.01) and val_depth(joints[0], joints[15], 0.04)
        val[0, 16] = val_length(joints[0], joints[16], lengths["nose-eye"], 0.02, 0.01) and val_depth(joints[0], joints[16], 0.04)
        # eye-ear: with this deviation down it will invalidate front view of face but make it robuster to occlusion. If front, eye detection should work
        val[15, 17] = val_length(joints[15], joints[17], lengths["eye-ear"], 0.03, 0.02) and val_depth(joints[15], joints[17], 0.07)
        val[16, 18] = val_length(joints[16], joints[18], lengths["eye-ear"], 0.03, 0.02) and val_depth(joints[16], joints[18], 0.07)

        # no depth because arm and leg very flexible
        # shoulder-elbow
        val[2, 3] = val_length(joints[2], joints[3], lengths["shoulder-elbow"], 0.03, 0.06)
        val[5, 6] = val_length(joints[5], joints[6], lengths["shoulder-elbow"], 0.03, 0.06)
        # elbow-wrist
        val[3, 4] = val_length(joints[3], joints[4], lengths["elbow-wrist"], 0.03, 0.03)
        val[6, 7] = val_length(joints[6], joints[7], lengths["elbow-wrist"], 0.03, 0.03)
        # hip-knee: very imprecise
        val[9, 10] = val_length(joints[9], joints[10], lengths["hip-knee"], 0.04, 0.07)
        val[12, 13] = val_length(joints[12], joints[13], lengths["hip-knee"], 0.04, 0.07)
        # knee-foot
        val[10, 11] = val_length(joints[10], joints[11], lengths["knee-foot"], 0.02, 0.05)
        val[13, 14] = val_length(joints[13], joints[14], lengths["knee-foot"], 0.02, 0.05)
        # foot-toe
        val[11, 22] = val_length(joints[11], joints[22], lengths["foot-toe"])
        val[14, 19] = val_length(joints[14], joints[19], lengths["foot-toe"], 0.05, 0.01)

        # 20, 21, 23, 24 emitted

        for i in range(25):
            # remove depth of supposedly incorrect joints
            if not (any(val[i]) or any(val[:, i])):
                val_joints[i, 2] = 0

            # remove joints with low confidence
            """if confidences[i] < 0:
                val_joints[i] = 0"""

        # reduce head to nose
        if all(val_joints[0] == 0):  # nose not detected
            if val_joints[15, 2] != 0 and val_joints[16, 2] != 0:   # both eyes validated
                val_joints[0] = (val_joints[15] + val_joints[16])/2
            elif val_joints[15, 2] != 0:    # one eye validated
                val_joints[0] = val_joints[15]
            elif val_joints[16, 2] != 0:    # one eye validated
                val_joints[0] = val_joints[16]
            elif val_joints[17, 2] != 0 and val_joints[18, 2] != 0:  # both ears validated
                val_joints[0] = (val_joints[17] + val_joints[18])/2
            elif val_joints[17, 2] != 0:  # one ear validated
                val_joints[0] = val_joints[17]
            elif val_joints[18, 2] != 0:  # one ear validated
                val_joints[0] = val_joints[18]
        elif val_joints[0, 2] == 0:  # nose not validated
            if val_joints[15, 2] != 0 and val_joints[16, 2] != 0:   # both eyes validated
                val_joints[0, 2] = (val_joints[15, 2] + val_joints[16, 2])/2
            elif val_joints[15, 2] != 0:    # one eye validated
                val_joints[0, 2] = val_joints[15, 2]
            elif val_joints[16, 2] != 0:    # one eye validated
                val_joints[0, 2] = val_joints[16, 2]
            elif val_joints[17, 2] != 0 and val_joints[18, 2] != 0:  # both ears validated
                val_joints[0, 2] = (val_joints[17, 2] + val_joints[18, 2])/2
            elif val_joints[17, 2] != 0:  # one ear validated
                val_joints[0, 2] = val_joints[17, 2]
            elif val_joints[18, 2] != 0:  # one ear validated
                val_joints[0, 2] = val_joints[18, 2]

        return val_joints


def stream(savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = float("inf"),
           resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 1, countdown: int = 0, translation=(0, 0, 0),
           show_rgb: bool = True, show_depth: bool = True, show_joints: bool = True,
           simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=False, duration=duration, playback_file=None,
                      resolution=resolution, fps=fps, flip=flip, countdown=countdown, translation=translation,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joints=show_joints,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


def playback(playback_file: str, savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = -1,
             resolution: tuple[int, int] = (480, 270), fps: int = 30, flip: int = 1, translation=(0, 0, 0),
             show_rgb: bool = True, show_depth: bool = True, show_joints: bool = True,
             simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=True, duration=duration, playback_file=playback_file,
                      resolution=resolution, fps=fps, flip=flip, countdown=0, translation=translation,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joints=show_joints,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


if __name__ == '__main__':
    cam_translation = (0, 1.5, 0)    # from (x,y,z) = (0,0,0) in m
    cam_rotation = (0, 0, 0)   # from pointing parallel to z-axis, (x-rotation [up/down], y-rotation [left/right], z-rotation/tilt [anti-clockwise, clockwise]), in radians

    # playback("test.bag", save_joints=True, savefile_prefix="vid", simulate_limbs = False, simulate_joints = False, simulate_joint_connections = False)
    # playback("test.bag", translation=cam_translation, simulate_joint_connections=False, simulate_joints=False)

    # stream(countdown=3, translation=cam_translation, rotate=2, show_rgb=True, show_depth=True, show_joints=True, simulate_joints=False, simulate_joint_connections=False, simulate_limbs=True)

    stream(translation=cam_translation, flip=0, show_rgb=True, show_depth=True, show_joints=True, simulate_joints=False, simulate_joint_connections=False, simulate_limbs=True)
