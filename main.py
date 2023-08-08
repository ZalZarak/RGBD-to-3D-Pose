import time
import cv2
import numpy as np
import pyrealsense2 as rs
from cfonts import render as render_text
import pybullet_simulation as sim
from helper import print_countdown
from old.main import visualize_points
from openpose_handler import OpenPoseHandler


class RGBDto3DPose:
    def __init__(self, playback: bool, duration: float, playback_file: str | None, resolution: tuple[int, int], fps: int, rotate: int, countdown: int,
                 savefile_prefix: str | None, save_joints: bool, save_bag: bool, show_rgb: bool, show_depth: bool, show_joint_video: bool):
        self.playback = playback
        self.duration = duration
        self.playback_file = playback_file
        self.resolution = resolution
        self.fps = fps
        self.rotate = rotate
        self.countdown = countdown
        self.save_prefix = savefile_prefix
        self.save_joints = save_joints
        self.save_bag = save_bag
        self.show_rgb = show_rgb
        self.show_depth = show_depth
        self.show_joint_video = show_joint_video

        self.use_openpose = save_joints or show_joint_video
        self.colorizer = rs.colorizer()  # Create colorizer object
        self.result_directory = "results/"
        self.postfix_bag = ".bag"
        self.postfix_joints = "_joints.npy"

        if playback and (duration is None or duration <= 0):
            self.duration = float("inf")

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
        self.joints = []


    def run(self):
        print("Use ESC to terminate, otherwise no files will be saved.")
        try:
            max_frames = int(self.duration * self.fps)  # frames = seconds * fps
        except OverflowError:
            max_frames = self.duration
        frame_counter = 0
        key = None

        self.prepare()

        try:
            while key != 27 and frame_counter < max_frames:
                self.process_frame()

                key = cv2.waitKey(1)
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()


    def prepare(self):
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
            config.enable_record_to_file(self.save_prefix + self.postfix_bag)
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

        if self.use_openpose:
            joints = self.openpose_handler.push_frame(color_image, self.show_joint_video)
            self.joints.append(joints)
        if self.show_rgb:
            cv2.imshow("RGB-Stream", color_image)
        if self.show_depth:
            cv2.imshow("Depth-Stream", depth_image)


def stream(savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = float("inf"),
           resolution: tuple[int, int] = (480, 270), fps: int = 30, rotate: int = 1, countdown: int = 3,
           show_rgb: bool = False, show_depth: bool = True, show_joint_video: bool = True):

    cl = RGBDto3DPose(playback=False, duration=duration, playback_file=None, resolution=resolution, fps=fps, rotate=rotate, countdown=countdown,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joint_video=show_joint_video)
    cl.run()


def playback(playback_file: str, savefile_prefix: str | None = None, save_joints: bool = False, save_bag: bool = False, duration: float = -1,
             resolution: tuple[int, int] = (480, 270), fps: int = 30, rotate: int = 1,
             show_rgb: bool = False, show_depth: bool = True, show_joint_video: bool = True):

    cl = RGBDto3DPose(playback=True, duration=duration, playback_file=playback_file, resolution=resolution, fps=fps, rotate=rotate, countdown=0,
                      savefile_prefix=savefile_prefix, save_joints=save_joints, save_bag=save_bag,
                      show_rgb=show_rgb, show_depth=show_depth, show_joint_video=show_joint_video)
    cl.run()


if __name__ == '__main__':
    playback("test.bag")
