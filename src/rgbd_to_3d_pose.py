import os
import pickle
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from math import sin, cos

import multiprocessing as mp

from src import helper, simulator, custom_3d_joint_transforms
from src.config import config
from src.openpose_handler import OpenPoseHandler


class RGBDto3DPose:
    def __init__(self, playback: bool, duration: float, playback_file: str | None,
                 resolution: tuple[int, int], fps: int, flip: int, countdown: int,
                 translation: (float, float, float), rotation: (float, float, float),
                 savefile_prefix: str | None, save_joints: bool, save_bag: bool,
                 show_rgb: bool, show_depth: bool, show_joints: bool, show_color_mask: bool,
                 simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool,
                 visual_preset: str, exposure: int, gain: int, laser_power: int, depth_units: float,
                 decimation_filter: bool, depth2disparity: bool, spatial_filter: bool,
                 temporal_filter: bool, disparity2depth: bool, hole_filling_filter: int,
                 joint_map: dict, connections_hr: list[tuple[str, str]], color_validation_joints_hr: list[str],
                 color_range: tuple[tuple[int, int, int], tuple[int, int, int]], lengths_hr: dict, depth_deviations_hr: dict,
                 search_areas_hr: dict, joint_3d_transforms: [str],
                 start_simulator=True, joints_sync=None, ready_sync=None, done_sync=None, new_joints_sync=None
                 ):
        """
        Receives streams from Intel RealSense Depth Camera or a playback file, pushes to OpenPose for Joint Positions,
        extracts 3D coordinates, validates and corrects them through different techniques. Starts Simulator as separate process and pushes
        3D coordinates of Joints to simulator.

        :param playback: If playback from a recorded bag instead of streaming from the camera.
        :param duration: How long to stream or playback, in seconds. Use -1 or float('inf') to not stop automatically after a certain period.
        :param playback_file: File to playback from. Ignored if playback == False.
        :param resolution: Camera resolution. Check documentation or the Intel RealSenseViewer for possible resolutions.
        :param fps: Configure the camera for this number of frames per second. Program can run with fewer fps (especially the Simulator).
        :param flip: Rotation of the camera in 90° steps. 0: no rotation. 1: 90° clockwise, 2: 180°, -1: 90° anti-clockwise.
                     In contrast to rotation, this is directly applied to the received image and pushed to OpenPose.
                     OpenPose should receive an upright image for the best results.
        :param countdown: Countdown until start of stream/playback, in seconds.
        :param translation: (x,y,z) translation to apply to 3D joints at the end, in meters.
        :param rotation: (x-axis, y-axis, z-axis) rotation to apply to 3D joints at the end, in radians.
        :param savefile_prefix: Prefix for saved files.
        :param save_joints: If 3D joints and the time they were received should be saved under savefile_prefix + "_joints.pkl".
        :param save_bag: If a bag of the color and depth stream should be saved under savefile_prefix + ".bag". Attention: No compression -> big files.
        :param show_rgb: Show RGB-Stream.
        :param show_depth: Show Depth-Stream.
        :param show_joints: Show joints and connections in both the RGB-Stream and the Depth-Stream.
        :param show_color_mask: Show a stream where all pixels outside the defined color range are black.
        :param simulate_limbs: Simulate limbs.
        :param simulate_joints: Simulate joints.
        :param simulate_joint_connections: Simulate connections between joints.
        :param visual_preset: Preset for the visual appearance of the depth stream.
        :param exposure: Camera exposure settings.
        :param gain: Camera gain settings.
        :param laser_power: Power of the camera's laser for depth sensing.
        :param depth_units: Units for depth measurement.
        :param decimation_filter: Whether to apply a decimation filter.
        :param depth2disparity: Convert depth to disparity.
        :param spatial_filter: Apply a spatial filter.
        :param temporal_filter: Apply a temporal filter.
        :param disparity2depth: Convert disparity back to depth.
        :param hole_filling_filter: Level of hole filling filter to apply.
        :param joint_map: Maps joint names to openpose indices.
        :param connections_hr: List of connections between joints with their names (human-readable)
        :param color_validation_joints_hr: List of joints to validate with color-range (human-readable)
        :param color_range: Color range to validate joints
        :param lengths_hr: Accepted length ranges for each connection
        :param depth_deviations_hr: Accepted maximum depth (z-coordinate) deviations for each connection
        :param search_areas_hr: Define the search area where a valid depth is searched for invalid joints for each joint. Format: [deviation, skip] in pixels.
        :param joint_3d_transforms: Names of custom transform functions for 3d-joints in custom_3d_joint_transforms.py.
                                    Will be applied in the given order in camera space after extracting 3d coordinates and performing validation.
                                    Visit custom_3d_joint_transforms.py for more information.
        :param start_simulator: If Simulator should start as separate process from here. Otherwise, Simulator should start a RGBDto3DPose process.
        :param joints_sync: If this is a subprocess, to exchange joints with Simulator, else None. mp.Array('f', np.zeros([25 * 3])), filled out automatically by Simulator
        :param ready_sync: If this is a subprocess, to communicate with Simulator when process is ready, else None. mp.Event, filled out automatically by Simulator
        :param done_sync: If this is a subprocess, to communicate with Simulator when process has finished, else None. mp.Event, filled out automatically by Simulator
        :param new_joints_sync: If this is a subprocess, to communicate with Simulator if there are new joints, else None mp.Event, filled out automatically by Simulator
        """

        self.playback = playback
        self.duration = float("inf") if duration is None or duration <= 0 else duration
        self.playback_file = playback_file

        translation = np.array(translation)
        rotation = np.array(rotation)
        rotx, roty, rotz = rotation
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
        rot_mat = x_rot_mat @ y_rot_mat @ z_rot_mat     # combine x,y,z rotations into one matrix

        # combine rotation and translation into one function
        if all(rotation == 0) and all(translation == 0):  # less calculation
            self.transform = lambda m: m
        elif all(rotation == 0):                             # less calculation
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
        self.colorizer = rs.colorizer()  # create colorizer object

        # real_to_camera_space_2d translates pixels from flipped real space to camera space. Real space is given by flip parameter.
        # camera_to_real_space_3d switches and mirrors dimension to adjust 3D coordinate from camera space to real space (x->right, y->up, z->back)
        # flip_rev is the reverse of flip
        if flip in [-4, 0, 4]:  # no rotation
            self.real_to_camera_space_2d = lambda p: np.array([round(p[0]), round(p[1])])  # (round(x), round(y))
            self.camera_to_real_space_3d = lambda c: (c[0], -c[1], c[2])
            self.flip_rev = 0
        elif flip in [-3, 1]:  # left rotation
            self.real_to_camera_space_2d = lambda p: np.array([resolution[0] - round(p[1]), round(p[0])])  # (resolution[0] - round(y), round(x))
            self.camera_to_real_space_3d = lambda c: (c[1], c[0], c[2])
            self.flip_rev = -1
        elif flip in [-2, 2]:  # 180° rotation
            self.real_to_camera_space_2d = lambda p: np.array([resolution[0] - round(p[0]), resolution[1] - round(p[1])])  # (resolution[0] - round(x), resolution[1] - round(y))
            self.camera_to_real_space_3d = lambda c: (-c[0], c[1], c[2])
            self.flip_rev = 2
        elif flip in [-1, 3]:  # right rotation
            self.real_to_camera_space_2d = lambda p: np.array([round(p[1]), resolution[1] - round(p[0])])  # (round(y), resolution[1] - round(x))
            self.camera_to_real_space_3d = lambda c: (-c[1], -c[0], c[2])
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
        if show_color_mask:
            cv2.namedWindow("Color-Mask-Stream", cv2.WINDOW_AUTOSIZE)

        self.simulate = simulate_limbs or simulate_joints or simulate_joint_connections
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections

        self.start_simulator = start_simulator          # if True and simulate, this will start Simulator, if false and simulate, Simulator will start this
        self.done_sync = done_sync if done_sync is not None else mp.Value('b', False)  # to communicate with simulator if one process ended
        self.ready_sync = ready_sync                    # to communicate when process is done initializing
        self.joints_sync = joints_sync                  # to forward joints to simulator
        self.new_joints_sync = new_joints_sync          # to communicate if there are new joint positions

        self.joints_save = []   # list to save time and 3D joints
        self.start_time = -1

        self.intrinsics = None
        self.pipeline = None
        self.openpose_handler: OpenPoseHandler = None

        # define filters
        self.align = rs.align(rs.stream.color)
        self.decimation_filter = rs.decimation_filter() if decimation_filter else helper.NoFilter()
        self.depth2disparity = rs.disparity_transform(True) if depth2disparity else helper.NoFilter()
        self.spatial_filter = rs.spatial_filter() if spatial_filter else helper.NoFilter()
        self.temporal_filter = rs.temporal_filter() if temporal_filter else helper.NoFilter()
        self.disparity2depth = rs.disparity_transform(False) if disparity2depth else helper.NoFilter()
        self.hole_filling_filter = rs.hole_filling_filter(hole_filling_filter) if hole_filling_filter >= 0 else helper.NoFilter()

        self.visual_preset = visual_preset
        self.exposure = exposure
        self.gain = gain
        self.laser_power = laser_power
        self.depth_units = depth_units

        for c1, c2 in connections_hr:
            if joint_map[c1] >= joint_map[c2]:
                raise ValueError(f"connections_list: for all tuples (a,b): joint_map[a] < joint_map[b], but joint_map[{c1}] >= joint_map[{c2}]")
        for c in color_validation_joints_hr:
            if c not in joint_map.keys():
                raise ValueError(f"color_validation_joints_hr: {c} not in joint_map")
        for a in color_range:
            for b in a:
                if not 0 <= b <= 255:
                    raise ValueError(f"color_range: values between 0 and 255")
        for k in lengths_hr.keys():
            k1, k2 = k.split("-")
            if joint_map[k1] >= joint_map[k2]:
                raise ValueError(f"lengths_hr: for all keys 'j1-j2': joint_map[j1] < joint_map[j2], but joint_map[{k1}] >= joint_map[{k2}]")
        for k in depth_deviations_hr.keys():
            k1, k2 = k.split("-")
            if joint_map[k1] >= joint_map[k2]:
                raise ValueError(f"depth_deviations_hr: for all keys 'j1-j2': joint_map[j1] < joint_map[j2], but joint_map[{k1}] >= joint_map[{k2}]")

        self.joint_map = joint_map
        self.connections_hr = connections_hr
        self.color_validation_joints_hr = color_validation_joints_hr
        self.color_range = np.array(color_range)
        self.lengths_hr = lengths_hr
        self.depth_deviations_hr = depth_deviations_hr
        self.search_areas_hr = search_areas_hr

        self.joint_map_rev = {v: k for k, v in joint_map.items()}

        self.connections = [(joint_map[c[0]], joint_map[c[1]]) for c in connections_hr]

        self.lengths = {}
        for k, v in lengths_hr.items():
            k1, k2 = k.split('-')
            self.lengths[(joint_map[k1], joint_map[k2])] = v

        self.depth_deviations = {}
        for k, v in depth_deviations_hr.items():
            k1, k2 = k.split('-')
            self.depth_deviations[(joint_map[k1], joint_map[k2])] = v

        self.search_areas = {}
        for k, v in search_areas_hr.items():
            self.search_areas[joint_map[k]] = helper.generate_base_search_area(v[0], v[1])

        self.connections_dict = {k: set() for k in range(26)}
        for a, b in self.connections:
            self.connections_dict[a].add(b)
            self.connections_dict[b].add(a)

        self.color_validation_joints = [joint_map[j] for j in color_validation_joints_hr]

        self.joint_3d_transforms = []
        for name in joint_3d_transforms:
            self.joint_3d_transforms.append(getattr(custom_3d_joint_transforms, name))

    def run(self):
        """
        Run the program as configured.

        :return: None
        """

        print("Use ESC to terminate, otherwise files might not be saved.")

        self.prepare()  # prepare everything needed

        key = None
        self.start_time = time.time()
        try:
            # while no windows closed, while not exceeded duration and while simulator is running (if started)
            while key != 27 and time.time() < self.start_time + self.duration and not self.done_sync.value:
                self.process_frame()
                key = cv2.waitKey(1)
        finally:
            if self.simulate:
                self.done_sync.value = True     # tell Simulator to stop
            cv2.destroyAllWindows()
            self.pipeline.stop()
            if self.save_joints:    # save joints
                with open(self.filename_joints, 'wb') as file:
                    pickle.dump(self.joints_save, file)
                print("Joints saved to:", self.filename_joints)

    def prepare(self):
        """
        Prepare everything needed (as configured).

        :return: None
        """

        # start Simulator
        if self.simulate and self.start_simulator:
            # create variables to exchange data with Simulator subprocess.
            self.done_sync = mp.Value('b', False)
            self.ready_sync = mp.Event()
            self.joints_sync = mp.Array('f', np.zeros([25 * 3]))
            self.new_joints_sync = mp.Event()
            # create and start Simulator subprocess
            simulator_process = mp.Process(target=simulator.run_as_subprocess,
                                           args=(self.joints_sync, self.ready_sync, self.done_sync, self.new_joints_sync,
                                                 self.simulate_limbs, self.simulate_joints, self.simulate_joint_connections))
            simulator_process.start()
            self.ready_sync.wait()   # wait until Simulator is ready

        # Initialize OpenPose Handler
        if self.use_openpose:
            self.openpose_handler = OpenPoseHandler()

        # Initialize the RealSense pipeline
        pipeline = rs.pipeline()
        rs_config = rs.config()

        if self.playback:
            # configure for playback from file.
            rs.config.enable_device_from_file(rs_config, file_name=self.playback_file)

        # Enable both depth and color streams
        rs_config.enable_stream(rs.stream.depth, self.resolution[0], self.resolution[1], rs.format.z16, self.fps)
        rs_config.enable_stream(rs.stream.color, self.resolution[0], self.resolution[1], rs.format.bgr8, self.fps)

        if self.save_bag:
            # setup to record to file
            rs_config.enable_record_to_file(self.filename_bag)
            pipeline_profile = pipeline.start(rs_config)   # start pipeline
            device = pipeline_profile.get_device()
            recorder = device.as_recorder()
            rs.recorder.pause(recorder)
            helper.print_countdown(self.countdown)
            rs.recorder.resume(recorder)
        else:
            if self.countdown > 0:
                helper.print_countdown(self.countdown)
            pipeline_profile = pipeline.start(rs_config)   # start pipeline

        self.intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        if not self.playback:
            # configure camera
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            for i in range(int(depth_sensor.get_option_range(rs.option.visual_preset).max)):
                if depth_sensor.get_option_value_description(rs.option.visual_preset, i) == self.visual_preset:
                    depth_sensor.set_option(rs.option.visual_preset, i)
            depth_sensor.set_option(rs.option.exposure, self.exposure)
            depth_sensor.set_option(rs.option.gain, self.gain)
            depth_sensor.set_option(rs.option.laser_power, self.laser_power)
            depth_sensor.set_option(rs.option.depth_units, self.depth_units)

        self.pipeline = pipeline

        if self.simulate and not self.start_simulator:
            # communicate that process is ready to simulator if this is subprocess
            self.ready_sync.set()

    def process_frame(self):
        """
        Process every frame as configured.

        :return: None
        """

        # get frames, images and time
        color_frame, color_image, depth_frame, depth_image, t = self.get_frames()

        if self.use_openpose:
            # get joints from OpenPose assuming there is only one person
            joints_2d, confidences, joint_image = self.openpose_handler.push_frame(color_image)

            if self.show_rgb:
                helper.show("RGB-Stream", joint_image if self.show_joints else color_image)
            if self.show_depth:
                helper.show("Depth-Stream", depth_image, joints_2d if self.show_joints else None, self.connections)

            joints_2d_camera_space = np.apply_along_axis(self.real_to_camera_space_2d, 1, joints_2d)    # translate from image space to camera space
            color_image_camera_space = np.rot90(color_image, k=self.flip_rev)  # rotate color_image back to camera space

            joints_3d = self.get_3d_joints(joints_2d_camera_space, confidences, depth_frame, color_image_camera_space)  # retrieve 3d coordinates for joints

            for f in self.joint_3d_transforms:
                # apply custom transforms
                joints_3d = f(joints_3d)

            for i in range(25):
                # set supposedly incorrect joints to zero
                if joints_3d[i, 2] == 0:
                    joints_3d[i] = 0

            joints_3d = np.apply_along_axis(self.camera_to_real_space_3d, 1, joints_3d)  # flip the coordinates into the right perspective
            joints_3d = np.apply_along_axis(self.transform, 1, joints_3d)         # apply translation/rotation

            if self.simulate:
                # set sync joints and new joints flag for Simulator
                self.joints_sync[:] = joints_3d.flatten()
                self.new_joints_sync.set()
            if self.save_joints:
                self.joints_save.append((t - self.start_time, joints_3d))
        else:
            if self.show_rgb:
                helper.show("RGB-Stream", color_image)
            if self.show_depth:
                helper.show("Depth-Stream", depth_image)
        if self.show_color_mask:
            helper.show_mask("Color-Mask-Stream", color_image, self.color_range)

    def get_frames(self) -> tuple[any, np.ndarray, any, np.ndarray, float]:
        """
        Get color_frame, color_image, depth_frame, depth_image from pipeline

        :return: color_frame, color_image, depth_frame, depth_image, time
        """

        # Wait for the next set of frames from the camera
        frames = self.pipeline.wait_for_frames()
        t = time.time()     # save the time the frames were received at
        frames = self.align.process(frames)

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # apply filters as defined
        depth_frame = self.decimation_filter.process(depth_frame)
        depth_frame = self.depth2disparity.process(depth_frame)
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
        depth_frame = self.disparity2depth.process(depth_frame)
        depth_frame = self.hole_filling_filter.process(depth_frame)
        color_frame = frames.get_color_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = self.colorizer.colorize(depth_frame)

        depth_frame = depth_frame.as_depth_frame()

        # Convert frames to numpy array to render image in opencv
        depth_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # flip to real space
        depth_image = np.rot90(depth_image, k=self.flip)
        color_image = np.rot90(color_image, k=self.flip)

        return color_frame, color_image, depth_frame, depth_image, t

    def get_3d_joints(self, joints_2d: np.ndarray, confidences: np.ndarray, depth_frame, color_image: np.ndarray) -> np.ndarray:
        """
        Get 3D coordinates of joints from their pixel and depth. Validate them through at least one connection by length and depth or color (as defined).
        Attempt correction of unvalidated joints by searching in the area around as defined for valid color (for certain defined joints) or
        for valid length and depth, so that at least one connection validates.

        :param joints_2d: 2d joints in camera space
        :param confidences: confidences of those joints
        :param depth_frame: depth frame in camera space
        :param color_image: color image in camera space
        :return: 3d joints in camera space
        """

        def validate_joint(connection: tuple[int, int]) -> bool:
            """
            Validate the connection by length and depth as defined. The joint coordinates from val_joints are used.
            :param connection: Connection to validate
            :return: True/False
            """

            # If one joint has no depth, it was not detected correctly
            if joints_3d[connection[0], 2] == 0 or joints_3d[connection[1], 2] == 0:
                return False

            # retrieve accepted bounds for length and depth
            l_min, l_max = self.lengths[connection]
            deviation = self.depth_deviations[connection]

            return ((l_min <= np.linalg.norm(joints_3d[connection[1]] - joints_3d[connection[0]]) <= l_max)  # length validation
                   and
                   (deviation < 0 or abs(joints_3d[connection[0], 2] - joints_3d[connection[1], 2]) <= deviation))  # depth validation


        joints_3d = np.zeros([joints_2d.shape[0], 3])
        # get 3d coordinates for all joints
        for i, (x, y) in enumerate(joints_2d):
            try:
                depth = depth_frame.get_distance(x, y)
                if depth > 0:
                    # get 3d coordinates
                    joints_3d[i] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)
                else:   # error in reading depth
                    joints_3d[i] = 0
            except RuntimeError:  # joint outside of picture
                pass

        val = np.zeros(25, dtype="bool")

        # Problem: Joints might be occluded, then 3d joint would be wrong
        # Idea: If 3d coordinate is wrong, then it's likely that the length of limbs should be incorrect or the depth deviation too high
        # If a color within the color range is detected where the joint is, it can be assumed that nothing occludes it, so that the 3d coordinate is correct.

        # validate each joint through at least one connection which is valid by length and depth
        for connection in self.connections:
            success = validate_joint(connection)
            # if validation is successful mark both joints as validated, otherwise leave them as is
            val[connection[0]] |= success
            val[connection[1]] |= success

        # search around the joint for the right color
        # if found, define the 3d coordinate of this joint by original x,y and the found depth
        # this will accept any positive depth without further validation
        # make sure to have very precise color range, e.g. detected pixels not belonging to corresponding limbs should be nearly zero
        for color_joint in self.color_validation_joints:     # iterate through corresponding, unvalidated joints
            if not val[color_joint]:
                # get the pixel of the current joint
                x, y = joints_2d[color_joint, 0], joints_2d[color_joint, 1]
                # generate search pixels around joint
                for x_search, y_search in helper.generate_search_pixels((x, y), color_joint, self.search_areas, self.resolution):
                    color = color_image[y_search, x_search]
                    if all(np.greater_equal(self.color_range[1], color)) and all(np.greater_equal(color, self.color_range[0])):  # if in color range
                        try:
                            depth = depth_frame.get_distance(x_search, y_search)    # get the depth at this pixel
                            if depth > 0:
                                # get coordinate of original x,y but with the new depth
                                joints_3d[color_joint] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)
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
                    # get the pixel of the current joint
                    x, y = joints_2d[i, 0], joints_2d[i, 1]
                    # generate search pixels around joint
                    for x_search, y_search in helper.generate_search_pixels((x, y), i, self.search_areas, self.resolution):    # for each pixel in search area
                        try:
                            depth = depth_frame.get_distance(x_search, y_search)    # get the depth at this pixel
                            if depth <= 0:
                                continue

                            # get coordinate of original x,y but with the new depth
                            joints_3d[i] = rs.rs2_deproject_pixel_to_point(intrin=self.intrinsics, pixel=(x, y), depth=depth)

                            # try if any connection validates successfully then break out and continue with the next joint
                            for connected_joint in self.connections_dict[i]:
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

        for i in range(25):
            # set z-coordinate of supposedly incorrect joints to zero
            if not val[i]:
                joints_3d[i, 2] = 0

        return joints_3d


def run():
    """
    Create RGBDto3DPose instance with defined config and run it as main process.
    :return: None
    """

    cl = RGBDto3DPose(**config["RGBDto3DPose"])
    cl.run()


def run_as_subprocess(simulate_limbs, simulate_joints, simulate_joint_connections, done_sync, ready_sync, joints_sync, new_joints_sync):
    """
    Create RGBDto3DPose instance with defined config and as subprocess of Simulator.

    :param simulate_limbs: If limbs are simulated
    :param simulate_joints: If joints are simulated
    :param simulate_joint_connections: If connections are simulated
    :param done_sync: Synchronized is done flag, mp.Event
    :param ready_sync: Synchronized is ready flag, mp.Event
    :param joints_sync: Synchronized joints, mp.Array('f', np.zeros([25 * 3]))
    :param new_joints_sync: Synchronized flag if there are new joints, mp.Event
    :return: None
    """

    config_RGBDto3DPose = config["RGBDto3DPose"]
    config_RGBDto3DPose["playback"] = False
    config_RGBDto3DPose["simulate_limbs"] = simulate_limbs
    config_RGBDto3DPose["simulate_joints"] = simulate_joints
    config_RGBDto3DPose["simulate_joint_connections"] = simulate_joint_connections
    config_RGBDto3DPose["start_simulator"] = False
    config_RGBDto3DPose["joints_sync"] = joints_sync
    config_RGBDto3DPose["ready_sync"] = ready_sync
    config_RGBDto3DPose["done_sync"] = done_sync
    config_RGBDto3DPose["new_joints_sync"] = new_joints_sync

    cl = RGBDto3DPose(**config_RGBDto3DPose)
    cl.run()


if __name__ == '__main__':
    cam_translation = (0, 1.5, 0)  # from (x,y,z) = (0,0,0) in m
    cam_rotation = (0, 0, 0)  # from pointing parallel to z-axis, (x-rotation [up/down], y-rotation [left/right], z-rotation/tilt [anti-clockwise, clockwise]), in radians

    run()
