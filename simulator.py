import atexit
from collections import deque
import math
import pickle
import time
from math import sqrt

import cv2
import numpy as np
import pybullet as p
import pybullet_data

import multiprocessing as mp


# from openpose_handler import OpenPoseHandler

pairs = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15),
         (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]

radii = {
    "head": 0.11,
    "neck": 0.06,
    "torso": 0.2,
    "arm": 0.05,
    "forearm": 0.04,
    "hand": 0.2,
    "thigh": 0.07,
    "leg": 0.06,
    "foot": 0.03
}

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

default_direction = np.array([0, 0, 1])

class Simulator:

    # TODO: Collision shapes: Dont forget to activate/deactivate
    def __init__(self, simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool,
                 joints_sync=None, ready_sync=None, done_sync=None, playback_file: str = None):
        self.joints_sync = joints_sync
        self.done_sync = done_sync
        self.playback_file = playback_file
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections

        self.limbs = {}

        # Connect to the PyBullet physics server
        physicsClient = p.connect(p.GUI)

        # Set the camera position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-10,
                                     cameraTargetPosition=[0, 0, 0])

        if simulate_limbs:
            # pregenerate geometry

            # head
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radii["head"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["head"] = body_id

            # neck
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["neck"], length=lengths["neck"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["neck"] = body_id

            # torso
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["torso"], length=lengths["torso"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["torso"] = body_id

            # armL
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["arm"], length=lengths["arm"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["armL"] = body_id

            # armR
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["arm"], length=lengths["arm"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["armR"] = body_id

            # forearmL
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["forearm"], length=lengths["forearm"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["forearmL"] = body_id

            # forearmR
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["forearm"], length=lengths["forearm"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["forearmR"] = body_id

            # handL
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radii["hand"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["handL"] = body_id

            # handR
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radii["hand"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["handR"] = body_id

            # thighL
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["thigh"], length=lengths["thigh"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["thighL"] = body_id

            # thighR
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["thigh"], length=lengths["thigh"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["thighR"] = body_id

            # legL
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["leg"], length=lengths["leg"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["legL"] = body_id

            # legR
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["leg"], length=lengths["leg"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["legR"] = body_id

            # footL
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["foot"], length=lengths["foot"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["footL"] = body_id

            # footR
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii["foot"], length=lengths["foot"])
            body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[0, -10, 0])
            self.limbs["footR"] = body_id

            for name in self.limbs.keys():
                p.changeVisualShape(self.limbs[name], -1, rgbaColor=[0, 0, 0, 0])

            self.limbs_directions = {key: [0, 0, 1] for key in self.limbs}

        if self.simulate_joints:
            self.points = {}
            for i in range(25):
                sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 0, 0])
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[0, 0, 0])
                self.points[i] = body_id
            self.connections = []

        if ready_sync is not None:
            ready_sync.set()

    def run_sync(self):
        try:
            while not self.done_sync.value:
                joints = np.array(self.joints_sync).reshape([25, 3])
                self.process_frame(joints)
            p.disconnect()
        finally:
            self.done_sync.value = True


    def run_playback(self, mode: int):
        """

        :param mode: 0: normal, 1: realtime, 2: step-by-step
        :return:
        """
        with open(self.playback_file, 'rb') as file:
            frames: [(float, np.ndarray)] = pickle.load(file)

        if mode == 1:
            frames = deque(frames)
            start = time.time()
            try:
                while True:
                    t, joints = frames.popleft()
                    while t < time.time() - start:
                        # t, joints = frames.pop()
                        t, joints = frames.popleft()
                    self.process_frame(joints)
            except IndexError:
                pass
        elif mode == 2:
            print("Press any key to simulate one frame. Press q to terminate")
            frames = deque(frames)
            try:
                while True:
                    key = input()
                    if key == "q":
                        break
                    t, joints = frames.popleft()
                    self.process_frame(joints)
            except IndexError:
                pass
        else:
            start = time.time()
            c = 0
            for _, joints in frames:
                self.process_frame(joints)
                c += 1
            print(f"fps: {c/(time.time()-start)}")
        p.disconnect()

    def process_frame(self, joints: np.ndarray):
        if self.simulate_joints:
            self.move_points(joints)
        if self.simulate_joint_connections:
            self.visualize_connections(joints)
        if self.simulate_limbs:
            self.move_limbs(convert_openpose_coords(joints))
        p.stepSimulation()

    def move_limbs(self, joints: dict[str, list[np.ndarray]]):
        for missing in self.limbs.keys() - joints.keys():
            p.changeVisualShape(self.limbs[missing], -1, rgbaColor=[0, 0, 0, 0])

        for limb, joint in joints.items():
            limb_id = self.limbs[limb]

            if limb in ["head", "handL", "handR"]:
                p.resetBasePositionAndOrientation(limb_id, joint[0], p.getQuaternionFromEuler([0, 0, 0]))
            else:
                coord1, coord2 = joint

                # Calculate midpoint
                midpoint = (coord1 + coord2) / 2

                direction = coord2 - coord1

                # Calculate the orientation quaternion
                rotation_axis = np.cross(default_direction, direction)
                rotation_angle = np.arccos(np.dot(default_direction, direction) / (np.linalg.norm(default_direction) * np.linalg.norm(direction)))
                """print(f"Input: {np.dot(default_direction, direction) / (np.linalg.norm(default_direction) * np.linalg.norm(direction))}")
                print(f"Angle: {rotation_angle}")"""
                orientation = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)

                # Update existing cylinder's properties
                p.resetBasePositionAndOrientation(limb_id, midpoint, orientation)

            p.changeVisualShape(limb_id, -1, rgbaColor=[0, 0, 0.9, 0.5])

    def move_points(self, joints):
        for point_id, point in enumerate(joints):
            id = self.points[point_id]
            if not point[2] == 0:
                x, z, y = point
                p.resetBasePositionAndOrientation(id, [x, y, z], p.getQuaternionFromEuler([0, 0, 0]))
                p.changeVisualShape(id, -1, rgbaColor=[0.9, 0, 0, 1.0])
            else:
                p.changeVisualShape(id, -1, rgbaColor=(0, 0, 0, 0))

    def visualize_connections(self, joints):
        for i in self.connections:
            p.removeUserDebugItem(self.connections[i])

        for i, connection in enumerate(pairs):
            point_id1, point_id2 = connection
            (x1, z1, y1), (x2, z2, y2) = joints[point_id1], joints[point_id2]

            # Retrieve the corresponding sphere IDs
            if y1 != 0 and y2 != 0:
                # Draw a line between the points
                self.connections.append(p.addUserDebugLine(lineFromXYZ=(x1, y1, z1), lineToXYZ=(x2, y2, z2), lineColorRGB=[0, 0, 0.9], lineWidth=5))


def convert_openpose_coords(coords: np.ndarray) -> dict[str, list[np.ndarray]]:
        ret = {}

        def add_to_ret(name: str, positions: list[int]):
            if all([coords[pos, 2] != 0 for pos in positions]):
                ret[name] = [np.array([coords[pos, 0], coords[pos, 2], coords[pos, 1]]) for pos in positions]

        add_to_ret("head", [0])
        add_to_ret("neck", [0, 1])
        add_to_ret("torso", [1, 8])
        add_to_ret("armR", [2, 3])
        add_to_ret("armL", [5, 6])
        add_to_ret("forearmR", [3, 4])
        add_to_ret("forearmL", [6, 7])
        add_to_ret("handR", [4])
        add_to_ret("handL", [7])
        add_to_ret("thighR", [9, 10])
        add_to_ret("thighL", [12, 13])
        add_to_ret("legR", [10, 11])
        add_to_ret("legL", [13, 14])
        add_to_ret("footR", [11, 22])
        add_to_ret("footL", [14, 19])

        return ret


def limb_coords_generator(joints: np.ndarray):
    tuples = [(0,), (0, 1), (1, 8), (2, 3), (5, 6), (3, 4), (6, 7), (4,), (7,), (9, 10), (12, 13), (10, 11), (13, 14), (11, 22), (14, 19)]
    names = ["head", "neck", "torso", "armR", "armL", "forearmR", "forearmL", "handR", "handL", "thighR", "thighL", "legR", "legL", "footR", "footL"]

    while len(tuples) > 0:
        yield names.pop(), list(joints.take(tuples.pop(), axis=0))


"""def visualize(con, ready):
    vis = Visualizer()
    ready.set()
    
    start = time.time()
    c = 0

    while time.time() - start < 10:
        joints = con.recv()
        # vis.visualize_points(joints)
        vis.move_limbs(convert_openpose_coords(joints))
        p.stepSimulation()
        c+=1
    print(f"fps:{c/10}")
    p.disconnect()"""


"""def visualize(joints_sync, ready, simulate_shape: bool, simulate_joints: bool, simulate_joint_connections: bool):
    vis = Simulator(joints_sync, ready, simulate_shape, simulate_joints, simulate_joint_connections)
    ready.set()

    start = time.time()
    c = 0

    while time.time() - start < 10:
        joints = np.array(joints_sync).reshape([25, 3])
        vis.move_points(joints)
        vis.move_limbs(convert_openpose_coords(joints))
        p.stepSimulation()
        c+=1
    print(f"fps:{c/10}")
    p.disconnect()"""


def simulate_sync(joints_sync, ready, done, simulate_shape: bool, simulate_joints: bool, simulate_joint_connections: bool):
    sim = Simulator(simulate_shape, simulate_joints, simulate_joint_connections, joints_sync, ready, done)
    sim.run_sync()


def simulate_playback(simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool, playback_file: str, mode: int):
    sim = Simulator(simulate_limbs, simulate_joints, simulate_joint_connections, playback_file=playback_file)
    sim.run_playback(mode)


def simulate_single(simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool, joints: list[np.ndarray]):
    sim = Simulator(simulate_limbs, simulate_joints, simulate_joint_connections)

    print("Press any key to simulate one frame. Press q to terminate")
    frames = deque(joints)
    try:
        while True:
            key = input()
            if key == "q":
                break
            joints = frames.popleft()
            sim.process_frame(joints)
    except IndexError:
        pass


if __name__ == '__main__':
    point_list1 = np.array([
        [0.00, 1.90, 1.00],  # Nose
        [0.00, 1.80, 1.00],  # Neck
        [-0.10, 1.80, 1.00],  # RightShoulder
        [-0.20, 1.30, 1.00],  # RightElbow
        [-0.30, 0.90, 1.00],  # RightWrist
        [0.10, 1.80, 1.00],  # LeftShoulder
        [0.20, 1.30, 1.00],  # LeftElbow
        [0.30, 0.90, 1.00],  # LeftWrist
        [0.00, 1.20, 1.00],  # MidHip
        [-0.10, 1.20, 1.00],  # RightHip
        [-0.10, 0.65, 1.00],  # RightKnee
        [-0.10, -0.01, 1.00],  # RightAnkle
        [0.10, 1.20, 1.00],  # LeftHip
        [0.10, 0.65, 1.00],  # LeftKnee
        [0.10, -0.01, 1.00],  # LeftAnkle
        [-0.02, 1.91, 1.00],  # RightEye
        [0.02, 1.91, 1.00],  # LeftEye
        [-0.04, 1.89, 1.00],  # RightEar
        [0.04, 1.89, 1.00],  # LeftEar
        [0.12, -0.01, 1.01],  # LeftBigToe
        [0.14, -0.01, 1.01],  # LeftSmallToe
        [0.11, -0.03, .99],  # LeftHeel
        [-0.12, - .01, 1.01],  # RightBigToe
        [- .14, - .01, 1.01],  # RightSmallToe
        [- .11, - .03, .99],  # RightHeel
    ])

    point_list2 = np.array([
        [0.00, 1.90, 1.00],  # Nose
        [0.00, 1.80, 1.00],  # Neck
        [-0.10, 1.80, 1.00],  # RightShoulder
        [-0.20, 1.30, 1.00],  # RightElbow
        [-0.30, 0.90, 1.00],  # RightWrist
        [0.10, 1.80, 1.00],  # LeftShoulder
        [0.20, 1.30, 1.00],  # LeftElbow
        [0.30, 0.90, 0.50],  # LeftWrist
        [0.00, 1.20, 1.00],  # MidHip
        [-0.10, 1.20, 1.00],  # RightHip
        [-0.10, 0.65, 1.00],  # RightKnee
        [-0.10, -0.01, 1.00],  # RightAnkle
        [0.10, 1.20, 1.00],  # LeftHip
        [0.10, 0.65, 1.00],  # LeftKnee
        [0.10, -0.01, 1.00],  # LeftAnkle
        [-0.02, 1.91, 1.00],  # RightEye
        [0.02, 1.91, 1.00],  # LeftEye
        [-0.04, 1.89, 1.00],  # RightEar
        [0.04, 1.89, 1.00],  # LeftEar
        [0.12, -0.01, 1.01],  # LeftBigToe
        [0.14, -0.01, 1.01],  # LeftSmallToe
        [0.11, -0.03, .99],  # LeftHeel
        [-0.12, - .01, 1.01],  # RightBigToe
        [- .14, - .01, 1.01],  # RightSmallToe
        [- .11, - .03, .99],  # RightHeel
    ])

    point_list3 = np.array([
        [0.00, 1.90, 1.00],  # Nose
        [0.00, 1.80, 1.00],  # Neck
        [-0.10, 1.80, 1.00],  # RightShoulder
        [-0.20, 1.30, 1.00],  # RightElbow
        [-0.30, 0.90, 1.00],  # RightWrist
        [0.10, 1.80, 1.00],  # LeftShoulder
        [0.20, 1.30, 1.00],  # LeftElbow
        [0.50, 0.90, 0.50],  # LeftWrist
        [0.00, 1.20, 1.00],  # MidHip
        [-0.10, 1.20, 1.00],  # RightHip
        [-0.10, 0.65, 1.00],  # RightKnee
        [-0.10, -0.01, 1.00],  # RightAnkle
        [0.10, 1.20, 1.00],  # LeftHip
        [0.10, 0.65, 1.00],  # LeftKnee
        [0.10, -0.01, 1.00],  # LeftAnkle
        [-0.02, 1.91, 1.00],  # RightEye
        [0.02, 1.91, 1.00],  # LeftEye
        [-0.04, 1.89, 1.00],  # RightEar
        [0.04, 1.89, 1.00],  # LeftEar
        [0.12, -0.01, 1.01],  # LeftBigToe
        [0.14, -0.01, 1.01],  # LeftSmallToe
        [0.11, -0.03, .99],  # LeftHeel
        [-0.12, - .01, 1.01],  # RightBigToe
        [- .14, - .01, 1.01],  # RightSmallToe
        [- .11, - .03, .99],  # RightHeel
    ])

    # simulate_single(True, True, True, [point_list1, point_list2, point_list3])
    simulate_playback(True, False, False, "test_joints.pkl", 1)
