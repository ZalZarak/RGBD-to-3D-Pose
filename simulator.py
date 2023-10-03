import atexit
from bisect import bisect_left
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

from config import config

# from openpose_handler import OpenPoseHandler

default_direction = np.array([0, 0, 1])


def is_joint_valid(joint: np.ndarray):
    return joint[1] != 0


class Simulator:

    # TODO: Collision shapes: Dont forget to activate/deactivate
    def __init__(self, simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool,
                 joint_map: dict, limbs: list[tuple[str, str]], radii: dict, lengths: dict,
                 joints_sync=None, ready_sync=None, done_sync=None, playback_file: str = None,
                 start_RGBDto3DPose=False):
        self.joints_sync = joints_sync
        self.done_sync = done_sync
        self.ready_sync = ready_sync
        self.playback_file = playback_file
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections
        self.limb_list = []
        self.move_in_physic_sim = True
        self.min_distance_to_move_outside_physic_sim = 0.1
        self.time_delta = 0.01
        for i in limbs:
            l = []
            for j in i:
                l.append(joint_map[j])
            self.limb_list.append(tuple(l))
        self.joint_list = []
        for i in self.limb_list:
            for j in i:
                self.joint_list.append(j)
        self.joint_list = set(self.joint_list)
        self.debug_lines = []

        # start main
        if start_RGBDto3DPose:
            self.done_sync = mp.Value('b', False)
            self.ready_sync = mp.Event()
            self.joints_sync = mp.Array('f', np.zeros([25 * 3]))

            from main import run_as_subprocess
            cl_process = mp.Process(target=run_as_subprocess,
                                    args=(simulate_limbs, simulate_joints, simulate_joint_connections, self.done_sync, self.ready_sync, self.joints_sync))
            cl_process.start()

        # Connect to the PyBullet physics server
        physicsClient = p.connect(p.GUI)
        p.setPhysicsEngineParameter(enableConeFriction=True)

        # Set the camera position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-10,
                                     cameraTargetPosition=[0, 0, 0])

        if simulate_limbs:
            self.limbs_pb = {}
            # pregenerate geometry
            n = 1
            for limb in limbs:
                if len(limb) == 1:  # if this limb has only one point, like head or hand, it's simulated with a sphere
                    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radii[limb[0]]*n, rgbaColor=[0, 0, 0, 0])
                    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radii[limb[0]] * n)
                    body_id = p.createMultiBody(baseMass=-1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=visual_shape, basePosition=[0, 0, 0])
                    self.limbs_pb[(joint_map[limb[0]],)] = body_id
                elif len(limb) == 2: # if this limb has two points, like head or hand, it's simulated with a cylinder
                    limb_str = f"{limb[0]}-{limb[1]}"
                    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii[limb_str]*n, length=lengths[limb_str]*n, rgbaColor=[0, 0, 0, 0])
                    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radii[limb_str] * n, height=lengths[limb_str] * n)
                    body_id = p.createMultiBody(baseMass=-1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=visual_shape, basePosition=[0, 0, 0])
                    if joint_map[limb[0]] >= joint_map[limb[1]]:
                        raise ValueError(f"limbs: for all tuples (a,b): joint_map[a] < joint_map[b], but joint_map[{limb[0]}] >= joint_map[{limb[1]}]")
                    self.limbs_pb[(joint_map[limb[0]], joint_map[limb[1]])] = body_id
                else:
                    raise ValueError(f"limbs: no connections between more then two joints: {limb}")

        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[255, 0, 0, 0.5])
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        body_id = p.createMultiBody(baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[0.5, 0.5, 1])

        """for body1 in self.limbs_pb.values():
            for body2 in self.limbs_pb.values():
                if body1 != body2:
                    p.setCollisionFilterPair(body1, body2, -1, -1, False)"""
        for body1 in self.limbs_pb.values():
            p.setCollisionFilterGroupMask(body1, -1, 1, 0)

        if self.simulate_joints:
            self.joints_pb = {}
            for j in self.joint_list:
                sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 0, 0])
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[0, 0, 0])
                self.joints_pb[j] = body_id

        if playback_file is not None and playback_file != "":
            with open(self.playback_file, 'rb') as file:
                self.frames: [(float, np.ndarray)] = list(pickle.load(file))

        if start_RGBDto3DPose:
            self.ready_sync.wait()
        elif playback_file is None:
            self.ready_sync.set()


    def run_sync(self):
        try:
            while not self.done_sync.value:
                self.process_frame_sync()
            p.disconnect()
        finally:
            self.done_sync.value = True

    def process_frame_sync(self):
        joints = np.array(self.joints_sync).reshape([25, 3])
        self.step(joints)

    def run_playback(self, mode: int):
        """

        :param mode: 0: normal, 1: realtime, 2: step-by-step
        :return:
        """
        if mode == 0:
            start = time.time()
            c = 0
            for _, joints in self.frames:
                self.step(joints)
                c += 1
            print(f"fps: {c / (time.time() - start)}")
        elif mode == 1:
            start = time.time()
            i = 0
            try:
                while True:
                    t, joints = self.frames[i]
                    i += 1
                    while t < time.time() - start:
                        # t, joints = frames.pop()
                        t, joints = self.frames[i]
                        i += 1
                    self.step(joints)
            except IndexError:
                pass
        elif mode == 2:
            print("Press any key to simulate one frame. Press q to terminate")
            i = 0
            try:
                while True:
                    key = input()
                    if key == "q":
                        break
                    t, joints = self.frames[i]
                    i += 1
                    self.step(joints)
            except IndexError:
                pass
        else:
            raise ValueError("Mode: 0, 1 or 2")
        p.disconnect()

    def process_frame_at_time(self, t: float):
        t = t % self.frames[-1][0]  # make it a loop
        i = bisect_left(self.frames, t, key=lambda j: j[0])
        _, joints = self.frames[i]
        self.step(joints)

    def step(self, joints: np.ndarray):
        joints = joints[:, [0, 2, 1]]   # adjust axis to fit pybullet axis
        if self.simulate_joints:
            self.move_points(joints)
        if self.simulate_joint_connections:
            self.visualize_connections(joints)
        if self.simulate_limbs:
            self.move_limbs(joints)
        p.stepSimulation()
        collisions = p.getContactPoints()
        if len(collisions) > 0:
            print("Collisions detected!")
        else:
            print("No collisions.")

    def move_limbs(self, joints: np.ndarray):
        for limb in self.limb_list:
            limb_pb = self.limbs_pb[limb]
            if all([is_joint_valid(joints[l]) for l in limb]):  # if all limb joints are valid
                if len(limb) == 1:  # if it's a sphere
                    midpoint = joints[limb[0]]
                    orientation = p.getQuaternionFromEuler([0, 0, 0])
                    # p.resetBasePositionAndOrientation(limb_pb, joints[limb[0]], p.getQuaternionFromEuler([0, 0, 0]))
                else:   # it's a cylinder
                    coord1, coord2 = joints[limb[0]], joints[limb[1]]

                    # Calculate midpoint
                    midpoint = (coord1 + coord2) / 2

                    direction = coord2 - coord1

                    # Calculate the orientation quaternion
                    rotation_axis = np.cross(default_direction, direction)
                    rotation_angle = np.arccos(np.dot(default_direction, direction) / (np.linalg.norm(default_direction) * np.linalg.norm(direction)))
                    orientation = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)

                if self.move_in_physic_sim:
                    current_midpoint, current_orientation = p.getBasePositionAndOrientation(limb_pb)
                    position_diff = np.array(midpoint) - np.array(current_midpoint)

                    if np.linalg.norm(position_diff) >= self.min_distance_to_move_outside_physic_sim:
                        p.resetBasePositionAndOrientation(limb_pb, midpoint, orientation)
                    else:
                        linear_vel = position_diff / self.time_delta

                        # Calculate angular velocity
                        orientation_diff = p.getDifferenceQuaternion(current_orientation, orientation)
                        axis, angle = p.getAxisAngleFromQuaternion(orientation_diff)
                        angular_vel = (np.array(axis) * angle) / self.time_delta

                        # Set the linear and angular velocities using p.resetBaseVelocity
                        p.resetBaseVelocity(limb_pb, linearVelocity=linear_vel, angularVelocity=angular_vel)
                else:
                    # Update existing cylinder's properties
                    p.resetBasePositionAndOrientation(limb_pb, midpoint, orientation)

                p.changeVisualShape(limb_pb, -1, rgbaColor=[0, 0, 0.9, 0.5])
            else:   # one limb joint is invalid
                p.changeVisualShape(limb_pb, -1, rgbaColor=[0, 0, 0, 0])

    def move_points(self, joints):
        for j in self.joint_list:
            joint_pb = self.joints_pb[j]
            if is_joint_valid(joints[j]):
                p.resetBasePositionAndOrientation(joint_pb, joints[j], p.getQuaternionFromEuler([0, 0, 0]))
                p.changeVisualShape(joint_pb, -1, rgbaColor=[0.9, 0, 0, 1.0])
            else:
                p.changeVisualShape(joint_pb, -1, rgbaColor=(0, 0, 0, 0))

    def visualize_connections(self, joints):
        for i in self.debug_lines:
            p.removeUserDebugItem(self.debug_lines[i])

        for limb in self.limb_list:

            if len(limb) == 2 and all([is_joint_valid(joints[l]) for l in limb]):   # iterate over valid connections
                # Draw a line between the points
                self.debug_lines.append(p.addUserDebugLine(lineFromXYZ=joints[limb[0]], lineToXYZ=joints[limb[1]], lineColorRGB=[0, 0, 0.9], lineWidth=5))


def simulate_sync(simulate_shape: bool = True, simulate_joints: bool = False, simulate_joint_connections: bool = False):
    joint_map = config["Main"]["joint_map"]
    limbs = config["Simulator"]["limbs"]
    radii = config["Simulator"]["radii"]
    lengths = config["Simulator"]["lengths"]
    config_RGBDto3DPose = config["Main"]

    sim = Simulator(simulate_shape, simulate_joints, simulate_joint_connections,
                    joint_map, limbs, radii, lengths, start_RGBDto3DPose=True, config_RGBDto3DPose=config_RGBDto3DPose)
    sim.run_sync()


def simulate_sync_as_subprocess(joints_sync, ready_sync, done_sync, simulate_shape: bool, simulate_joints: bool, simulate_joint_connections: bool):
    joint_map = config["Main"]["joint_map"]
    limbs = config["Simulator"]["limbs"]
    radii = config["Simulator"]["radii"]
    lengths = config["Simulator"]["lengths"]

    sim = Simulator(simulate_shape, simulate_joints, simulate_joint_connections,
                    joint_map, limbs, radii, lengths,
                    joints_sync, ready_sync, done_sync)
    sim.run_sync()


def simulate_playback(simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool, playback_file: str, mode: int):
    joint_map = config["Main"]["joint_map"]
    limbs = config["Simulator"]["limbs"]
    radii = config["Simulator"]["radii"]
    lengths = config["Simulator"]["lengths"]

    sim = Simulator(simulate_limbs, simulate_joints, simulate_joint_connections,
                    joint_map, limbs, radii, lengths, playback_file=playback_file)
    sim.run_playback(mode)


def simulate_single(simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool, joints: list[np.ndarray]):
    joint_map = config["Main"]["joint_map"]
    limbs = config["Simulator"]["limbs"]
    radii = config["Simulator"]["radii"]
    lengths = config["Simulator"]["lengths"]

    sim = Simulator(simulate_limbs, simulate_joints, simulate_joint_connections,
                    joint_map, limbs, radii, lengths)

    print("Press any key to simulate one frame. Press q to terminate")
    frames = deque(joints)
    try:
        while True:
            key = input()
            if key == "q":
                break
            joints = frames.popleft()
            sim.step(joints)
    except IndexError:
        pass


if __name__ == '__main__':
    # simulate_single(True, True, True, [point_list1, point_list2, point_list3])
    simulate_playback(True, False, False, "test_joints.pkl", 1)
    #simulate_sync()