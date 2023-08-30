import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from cfonts import render as render_text

import helper
import pybullet_simulation as sim
from helper import print_countdown, draw_pixel_grid
from old.main import visualize_points
from openpose_handler import OpenPoseHandler
from simulator import simulate_sync

import multiprocessing as mp

lengths = {
    "neck": 0.2,
    "torso": 0.6,
    "arm": 0.35,
    "forearm": 0.30,
    "thigh": 0.5,
    "leg": 0.45,
    "foot": 0.15,
    "shoulder": 0.22,
    "hip": 0.13,
    "head": 0.8
}


class RGBDto3DPose:
    count = 0

    def __init__(self, playback: bool, duration: float, playback_file: str | None, resolution: tuple[int, int], fps: int, rotate: int, countdown: int,
                 savefile_prefix: str | None, save_joints: bool, save_bag: bool, show_rgb: bool, show_depth: bool, show_joint_video: bool,
                 simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool):
        self.playback = playback
        self.duration = duration
        self.playback_file = playback_file
        self.resolution = resolution
        self.fps = fps
        self.rotate = rotate
        self.countdown = countdown
        self.save_joints = save_joints
        self.save_bag = save_bag
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.show_joint_video = show_joint_video

        self.use_openpose = save_joints or show_joint_video
        self.colorizer = rs.colorizer()  # Create colorizer object
        self.filename_bag = savefile_prefix + ".bag"
        self.filename_joints = savefile_prefix + "_joints.pkl"

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

        if playback and (duration is None or duration <= 0):
            self.duration = float("inf")

        # define (inverse) rotation function to call deprojection at the correct pixel
        if rotate in [-4, 0, 4]:  # no rotation
            self.inverse_rotation = lambda x, y: (round(x), round(y))
            self.rotate_3d_coord = lambda c: (c[0], c[1], c[2])
        elif rotate in [-3, 1]:  # left rotation
            self.inverse_rotation = lambda x, y: (resolution[0] - round(y), round(x))
            self.rotate_3d_coord = lambda c: (c[1], c[0], c[2])
        elif rotate in [-2, 2]:  # 180° rotation
            self.inverse_rotation = lambda x, y: (resolution[0] - round(x), resolution[1] - round(y))
            self.rotate_3d_coord = lambda c: (c[0], c[1], c[2])
        elif rotate in [-1, 3]:  # right rotation
            self.inverse_rotation = lambda x, y: (round(y), resolution[1] - round(x))
            self.rotate_3d_coord = lambda c: (c[1], c[0], c[2])
        else:
            raise ValueError("Rotation should be in range [-4, 4]")

        if (save_bag or save_joints) and (savefile_prefix is None or savefile_prefix == ""):
            raise ValueError("Provide prefix for saving files")
        if playback and (playback_file is None or playback_file == ""):
            raise ValueError("Provide playback file")

        if show_rgb:
            cv2.namedWindow("RGB-Stream", cv2.WINDOW_AUTOSIZE)
        if show_depth:
            cv2.namedWindow("Depth-Stream", cv2.WINDOW_AUTOSIZE)
        if show_joint_video:
            cv2.namedWindow("Joint-Stream", cv2.WINDOW_AUTOSIZE)

        self.intrinsics = None
        self.pipeline = None
        self.openpose_handler: OpenPoseHandler = None

        self.simulate = simulate_limbs or simulate_joints or simulate_joint_connections
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections
        self.done = None
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
            while key != 27 and frame_counter < max_frames:
                self.process_frame()

                key = cv2.waitKey(1)
        finally:
            if self.done is not None:
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
            self.done = mp.Value('b', False)
            self.joints = mp.Array('f', np.zeros([25 * 3]))
            p = mp.Process(target=simulate_sync, args=(self.joints, ready, self.done, self.simulate_limbs, self.simulate_joints, self.simulate_joint_connections))
            p.start()
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
            pipeline_profile = pipeline.start(config)

        self.intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.pipeline = pipeline

    def process_frame(self):
        # Wait for the next set of frames from the camera
        frames = self.pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = self.colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.rot90(depth_image, k=self.rotate)
        color_image = np.rot90(color_image, k=self.rotate)

        if self.show_rgb:
            cv2.imshow("RGB-Stream", draw_pixel_grid(color_image))
        if self.show_depth:
            cv2.imshow("Depth-Stream", draw_pixel_grid(depth_image))
        if self.use_openpose:
            # get joints from OpenPose. Assume there is only one person
            res = self.openpose_handler.push_frame(color_image, self.show_joint_video)[0]
            joints, confidences = res[:, :2], res[:, 2]
            for i, c in enumerate(confidences):
                if c < 0.85:
                    joints[i] = (0, 0)
            joints = self.get_3d_coords(joints, depth_frame)
            joints_val = self.validate_joints(joints, confidences)

            if self.simulate:
                # self.main_conn.send(joints_val)
                self.joints[:] = joints_val.flatten()
            if self.save_joints:
                self.joints_save.append((time.time() - self.start_time, joints_val))

            """if self.count == 10:
                # visualize_points(joints_val, OpenPoseHandler.pairs, joints)
                visualize(joints, OpenPoseHandler.pairs)
            self.count += 1"""

    def get_3d_coords(self, joints: np.ndarray, depth_frame) -> np.ndarray:
        depths = depth_frame.as_depth_frame()
        coords = np.zeros([joints.shape[0], 3])

        for i, (x, y) in enumerate(joints):
            try:
                x, y = self.inverse_rotation(x, y)
                depth = depths.get_distance(x, y)

                # get 3d coordinates and reorder them from y,x,z to x,y,z
                coord = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)
                coords[i] = self.rotate_3d_coord(coord)  # coord[1], coord[0], coord[2]
            except RuntimeError:  # joint outside of picture
                pass

        return coords

    def validate_joints(self, joints: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        def val_length(joint1: np.ndarray, joint2: np.ndarray, expected_length: float, deviation: float = 0.1) -> bool:
            return expected_length - deviation <= np.linalg.norm(joint2 - joint1) <= expected_length + deviation

        def val_depth(joint1: np.ndarray, joint2: np.ndarray, deviation: float = 0.2) -> bool:
            return abs(joint1[2] - joint2[2]) <= deviation

        val = np.zeros((25, 25), dtype="bool")

        val_joints = np.copy(joints)

        # If depth is wrong, then the length of limbs should be incorrect or the depth deviation is too high

        # The upper body should have correct length and have approximately the same depth
        # torso
        val[1, 8] = val[8, 1] = val_length(joints[1], joints[8], lengths["torso"]) and val_depth(joints[1], joints[8])
        # neck
        val[0, 1] = val[1, 0] = val_length(joints[1], joints[0], lengths["neck"]) and val_depth(joints[1], joints[0])
        # shoulderL
        val[1, 2] = val[2, 1] = val_length(joints[1], joints[2], lengths["shoulder"]) and val_depth(joints[1], joints[2])
        # shoulderR
        val[1, 5] = val[5, 1] = val_length(joints[1], joints[5], lengths["shoulder"]) and val_depth(joints[1], joints[5])
        # hipL
        val[8, 9] = val[9, 8] = val_length(joints[8], joints[9], lengths["hip"]) and val_depth(joints[8], joints[9])
        # hipR
        val[8, 12] = val[12, 8] = val_length(joints[8], joints[12], lengths["hip"]) and val_depth(joints[8], joints[12])

        # head
        for i, j in [(0, 15), (0, 16), (15, 17), (16, 17)]:  # half head length
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["head"] / 2, 0.4) and val_depth(joints[8], joints[12], 0.4)
        for i, j in [(0, 17), (0, 18)]:  # head length
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["head"], 0.4) and val_depth(joints[8], joints[12], 0.4)
        for i, j in [(17, 18)]:  # double head length
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["head"] * 2, 0.6) and val_depth(joints[8], joints[12], 0.6)

        # no depth because arm and leg very flexible
        # upper arm
        for i, j in [(2, 3), (5, 6)]:
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["arm"])
        # lower arm
        for i, j in [(3, 4), (6, 7)]:
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["forearm"])
        # thigh
        for i, j in [(9, 10), (12, 13)]:
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["thigh"])
        # leg
        for i, j in [(10, 11), (13, 14)]:
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["leg"])
        # foot
        for i, j in [(11, 22), (14, 19)]:
            val[i, j] = val[j, i] = val_length(joints[i], joints[j], lengths["arm"])
        # 20, 21, 23, 24 emitted

        for i in range(25):
            # remove depth of supposedly incorrect joints
            if not (any(val[i]) or any(val[:, i])):
                val_joints[i, 2] = 0

            # remove joints with low confidence
            if confidences[i] < 0.5:
                val_joints[i] = 0

        return val_joints


def stream(savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = float("inf"),
           resolution: tuple[int, int] = (480, 270), fps: int = 30, rotate: int = 1, countdown: int = 3,
           show_rgb: bool = False, show_depth: bool = True, show_joint_video: bool = True,
           simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=False, duration=duration, playback_file=None, resolution=resolution, fps=fps, rotate=rotate, countdown=countdown,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joint_video=show_joint_video,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


def playback(playback_file: str, savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = -1,
             resolution: tuple[int, int] = (480, 270), fps: int = 30, rotate: int = 1,
             show_rgb: bool = False, show_depth: bool = True, show_joint_video: bool = True,
             simulate_limbs: bool = True, simulate_joints: bool = True, simulate_joint_connections: bool = True):
    cl = RGBDto3DPose(playback=True, duration=duration, playback_file=playback_file, resolution=resolution, fps=fps, rotate=rotate, countdown=0,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joint_video=show_joint_video,
                      simulate_limbs=simulate_limbs, simulate_joints=simulate_joints, simulate_joint_connections=simulate_joint_connections)
    cl.run()


if __name__ == '__main__':
    playback("test.bag", save_joints=True, savefile_prefix="vid", simulate_limbs = False, simulate_joints = False, simulate_joint_connections = False)
    # playback("test.bag")

    # mmlab
