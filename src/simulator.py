from bisect import bisect_left
import pickle
import time

import numpy as np
import pybullet as p

import multiprocessing as mp

from src.config import config


default_direction = np.array([0, 0, 1])


def is_joint_valid(joint: np.ndarray):
    return joint[1] != 0


class Simulator:

    # TODO: Collision shapes: Dont forget to activate/deactivate
    def __init__(self, start_pybullet: bool, playback: bool, playback_file: str | None, playback_mode: int, frame_duration: float,
                 simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool,
                 move_in_physic_sim: bool, min_distance_to_move_outside_physic_sim: float, time_delta_move_in_physic_sim: float,
                 joint_map: dict, limbs: list[tuple[str, str]], radii: dict, lengths: dict,
                 as_subprocess=False, joints_sync=None, ready_sync=None, done_sync=None, new_joints_sync=None):
        """
        Simulates a humanoid made up of spheres and cylinders from 3D joint positions. Either receives those synchronously from a
        running RGBDto3DPose process or reads them from a file.
        Current joint positions are stored in self.joints.

        :param start_pybullet: If this process should start PyBullet. True if standalone, False if integrated into another program which uses PyBullet.
        :param playback: If Simulator should read joints from provided file.
        :param playback_file: File to read joints from
        :param playback_mode: 0: normal mode, simulate each frame for [frame_duration] sec.
                              1: Real-Time, simulate each frame at time defined in the saved timesteps. Might skip frames.
                              2: Step-by-step, press any key to simulate next frame, "q" to quit.
        :param frame_duration: Duration for each frame. Applies only for normal playback mode.
        :param simulate_limbs: If limbs should be simulated.
        :param simulate_joints: If joint should be simulated.
        :param simulate_joint_connections: If connections between joint should be simulated.
        :param move_in_physic_sim: If limbs, whose distance between old and new position is less than [min_distance_to_move_outside_physic_sim] meters
                                   should be moved using physic simulation. If false, limbs will "teleport" to the new position.
                                   Must be True for collision detection to work.
        :param min_distance_to_move_outside_physic_sim: Minimal distance to move limb outside of physic simulation. Applies only if [move_in_physic_sim] is True.
                                                        Collision detection will only work if moved inside physic simulation.
                                                        Moving limbs over greater distance inside physic simulation takes time.
                                                        --> Value that worked well in practice: ~ 0.05
        :param time_delta_move_in_physic_sim: "Speed" of moving limbs in physics sim. Low values move limbs faster but collision detection is less precise.
                                              -> Value that worked well in practice: ~ 0.01
        :param joint_map: Maps joint names to openpose indices.
        :param limbs: List of limbs as lists of the joints they are made up from,
        :param radii: Radii of limbs
        :param lengths: lengths of limbs (for those consisting of two joints)
        :param as_subprocess: If this is a subprocess of RGBDto3DPose. If false, RGBDto3DPose is a subprocess of Simulator.
        :param joints_sync: If this is a subprocess, to exchange joints with RGBDto3DPose, else None. mp.Array('f', np.zeros([25 * 3])), filled out automatically by RGBDto3DPose
        :param ready_sync: If this is a subprocess, to communicate with RGBDto3DPose when process is ready, else None. mp.Event, filled out automatically by RGBDto3DPose
        :param done_sync: If this is a subprocess, to communicate with RGBDto3DPose when process has finished, else None. mp.Event, filled out automatically by RGBDto3DPose
        :param new_joints_sync: If this is a subprocess, to communicate with RGBDto3DPose if there are new joints, else None mp.Event, filled out automatically by RGBDto3DPose
        """

        self.playback = playback and not as_subprocess
        self.playback_file = playback_file
        self.playback_mode = playback_mode
        self.frame_duration = frame_duration
        self.simulate_limbs = simulate_limbs
        self.simulate_joints = simulate_joints
        self.simulate_joint_connections = simulate_joint_connections
        self.move_in_physic_sim = move_in_physic_sim
        self.min_distance_to_move_outside_physic_sim = min_distance_to_move_outside_physic_sim
        self.time_delta = time_delta_move_in_physic_sim
        self.limb_list = []
        # TODO valid/invalid ?
        self.limb_list_valid = []   # valid limbs of last processed frame
        self.limb_list_invalid = []  # invalid limbs of last processed frame
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
        self.joints = None  # stores current joints with axis in pybullet format
        self.joints_sync = joints_sync
        self.done_sync = done_sync
        self.ready_sync = ready_sync
        self.new_joints_sync = new_joints_sync

        # start RGBDto3DPose
        if not as_subprocess and not playback:
            self.done_sync = mp.Value('b', False)
            self.ready_sync = mp.Event()
            self.joints_sync = mp.Array('f', np.zeros([25 * 3]))
            self.new_joints_sync = mp.Event()

            from . import rgbd_to_3d_pose
            cl_process = mp.Process(target=rgbd_to_3d_pose.run_as_subprocess,
                                    args=(simulate_limbs, simulate_joints, simulate_joint_connections, self.done_sync, self.ready_sync, self.joints_sync, self.new_joints_sync))
            cl_process.start()

        # Connect to the PyBullet physics server
        if start_pybullet:
            p.connect(p.GUI)
        p.setPhysicsEngineParameter(enableConeFriction=True)

        # Set the camera position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-10,
                                     cameraTargetPosition=[0, 0, 0])

        self.limbs_pb = {}  # maps config ids of limbs to pybullet object ids
        if simulate_limbs:
            # pregenerate geometry
            for limb in limbs:
                if len(limb) == 1:  # if this limb has only one point, like head or hand, it's simulated with a sphere
                    visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radii[limb[0]], rgbaColor=[0, 0, 0, 0])
                    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radii[limb[0]])
                    body_id = p.createMultiBody(baseMass=-1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[0, 0, 0])
                    self.limbs_pb[(joint_map[limb[0]],)] = body_id
                elif len(limb) == 2: # if this limb has two points, like head or hand, it's simulated with a cylinder
                    limb_str = f"{limb[0]}-{limb[1]}"
                    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radii[limb_str], length=lengths[limb_str], rgbaColor=[0, 0, 0, 0])
                    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radii[limb_str], height=lengths[limb_str])
                    body_id = p.createMultiBody(baseMass=-1, baseVisualShapeIndex=visual_shape, baseCollisionShapeIndex=collision_shape, basePosition=[0, 0, 0])
                    if joint_map[limb[0]] >= joint_map[limb[1]]:
                        raise ValueError(f"limbs: for all tuples (a,b): joint_map[a] < joint_map[b], but joint_map[{limb[0]}] >= joint_map[{limb[1]}]")
                    self.limbs_pb[(joint_map[limb[0]], joint_map[limb[1]])] = body_id
                else:
                    raise ValueError(f"limbs: no connections between more then two joints: {limb}")

        # set collision filter so that limbs don't collide
        for body1 in self.limbs_pb.values():
            p.setCollisionFilterGroupMask(body1, -1, 1, 0)

        if self.simulate_joints:
            self.joints_pb = {}
            for j in self.joint_list:
                sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 0, 0])
                body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[0, 0, 0])
                self.joints_pb[j] = body_id

        if playback:
            try:
                with open(self.playback_file, 'rb') as file:
                    self.frames: [(float, np.ndarray)] = list(pickle.load(file))
            except FileNotFoundError:
                raise FileNotFoundError("Provide playback file")

        if not as_subprocess and not playback:
            # Wait for RGBDto3DPose to be ready
            self.ready_sync.wait()
        elif as_subprocess:
            # Communicate to RGBDto3DPose that Simulator is ready
            self.ready_sync.set()

    def run(self):
        """
        Run Simulator as configured.
        :return: None
        """

        if self.playback:
            self.run_playback()
        else:
            self.run_sync()

    def run_sync(self):
        """
        Run Simulator as configured as main process with RGBDto3DPose as subprocess.
        :return: None
        """

        try:
            while not self.done_sync.value:
                self.new_joints_sync.wait()
                self.new_joints_sync.clear()
                self.process_frame_sync()
            p.disconnect()
        finally:
            self.done_sync.value = True

    def process_frame_sync(self):
        """
        Process one frame. Get the joints from RGBDto3DPose.
        :return: None
        """

        joints = np.array(self.joints_sync).reshape([25, 3])
        self.step(joints)

    def run_playback(self):
        """
        Run simulation of playback file in the configured mode. Loops through the playback.
        :return: None
        """

        if self.playback_mode == 0:
            start = time.time()
            c = 0
            for _, joints in self.frames:
                self.step(joints)
                c += 1
            time.sleep(self.frame_duration)
            print(f"fps: {c / (time.time() - start)}")
        elif self.playback_mode == 1:
            start = time.time()
            try:
                while True:
                    self.process_frame_at_time(time.time()-start)
            except IndexError:
                pass
        elif self.playback_mode == 2:
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

    last_frame = -1
    def process_frame_at_time(self, t: float):
        """
        Simulate the frame whose timestamp is the nearest (bigger) to t. Loops through the playback.

        :param t: time of frame to process.
        :return: None
        """

        t = t % self.frames[-1][0]  # make it a loop
        i = bisect_left(self.frames, t, key=lambda j: j[0])

        # don't move if the frame is the same
        if self.last_frame != i:
            _, joints = self.frames[i]
            self.step(joints)
            self.last_frame = i

    def step(self, joints: np.ndarray):
        """
        Perform one simulation step. Save joints in self.joints.
        :param joints: New joint positions
        :return: None
        """

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(False))

        joints = joints[:, [0, 2, 1]]   # adjust axis to fit pybullet axis
        self.joints = joints.copy()
        if self.simulate_joints:
            self.move_points(joints)
        if self.simulate_joint_connections:
            self.visualize_connections(joints)
        if self.simulate_limbs:
            self.move_limbs(joints)
        p.stepSimulation()

        # Reset limb velocities to 0 prevent them from moving with potential additional simulation steps.
        if self.simulate_limbs:
            for limb_pb in self.limbs_pb.values():
                p.resetBaseVelocity(limb_pb, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(True))
        time.sleep(0.001)

    def move_limbs(self, joints: np.ndarray):
        """
        Move the limbs to the new positions. Depending on distance between new position and old position and config,
        move either with physic simulation or without.

        :param joints: new joint positions
        :return: None
        """

        self.limb_list_valid, self.limb_list_invalid = [], []
        for limb in self.limb_list:
            limb_pb = self.limbs_pb[limb]
            if all([is_joint_valid(joints[l]) for l in limb]):  # if all limb joints are valid
                self.limb_list_valid.append(limb)
                p.setCollisionFilterGroupMask(limb_pb, -1, 1, 0)    # activate collision
                if len(limb) == 1:  # if it's a sphere
                    midpoint = joints[limb[0]]
                    orientation = p.getQuaternionFromEuler([0, 0, 0])
                else:   # if it's a cylinder
                    coord1, coord2 = joints[limb[0]], joints[limb[1]]

                    midpoint = (coord1 + coord2) / 2

                    # Calculate the orientation quaternion
                    direction = coord2 - coord1
                    rotation_axis = np.cross(default_direction, direction)
                    rotation_angle = np.arccos(np.dot(default_direction, direction) / (np.linalg.norm(default_direction) * np.linalg.norm(direction)))
                    orientation = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)

                if self.move_in_physic_sim:
                    current_midpoint, current_orientation = p.getBasePositionAndOrientation(limb_pb)
                    position_diff = np.array(midpoint) - np.array(current_midpoint)

                    if np.linalg.norm(position_diff) >= self.min_distance_to_move_outside_physic_sim:   # if distance position difference to big
                        p.resetBasePositionAndOrientation(limb_pb, midpoint, orientation)   # move outside physic simulation
                    else:
                        linear_vel = position_diff / self.time_delta

                        # Calculate angular velocity
                        orientation_diff = p.getDifferenceQuaternion(current_orientation, orientation)
                        axis, angle = p.getAxisAngleFromQuaternion(orientation_diff)
                        angular_vel = (np.array(axis) * angle) / self.time_delta

                        # Set the linear and angular velocities and move inside physic simulation
                        p.resetBaseVelocity(limb_pb, linearVelocity=linear_vel, angularVelocity=angular_vel)
                else:
                    # move outside physic simulation
                    p.resetBasePositionAndOrientation(limb_pb, midpoint, orientation)

                p.changeVisualShape(limb_pb, -1, rgbaColor=[0, 0, 0.9, 0.5])    # make it visible
            else:   # one limb joint is invalid
                self.limb_list_invalid.append(limb)
                p.changeVisualShape(limb_pb, -1, rgbaColor=[0, 0, 0, 0])        # make invisible
                p.setCollisionFilterGroupMask(limb_pb, -1, 0, 0)                # deactivate collision

    def move_points(self, joints):
        """
        Move the points representing the joints to the new positions.
        :param joints: new joint positions
        :return: None
        """

        for j in self.joint_list:
            joint_pb = self.joints_pb[j]
            if is_joint_valid(joints[j]):
                p.resetBasePositionAndOrientation(joint_pb, joints[j], p.getQuaternionFromEuler([0, 0, 0]))
                p.changeVisualShape(joint_pb, -1, rgbaColor=[0.9, 0, 0, 1.0])   # make it visible
            else:
                p.changeVisualShape(joint_pb, -1, rgbaColor=(0, 0, 0, 0))       # make it invisible

    def visualize_connections(self, joints):
        """
        Visualize joint connections as debug lines
        :param joints: new joint positions
        :return: None
        """

        for i in self.debug_lines:
            p.removeUserDebugItem(self.debug_lines[i])

        for limb in self.limb_list:

            if len(limb) == 2 and all([is_joint_valid(joints[l]) for l in limb]):   # iterate over valid connections
                # Draw a line between the points
                self.debug_lines.append(p.addUserDebugLine(lineFromXYZ=joints[limb[0]], lineToXYZ=joints[limb[1]], lineColorRGB=[0, 0, 0.9], lineWidth=5))


def run():
    """
    Create Simulator instance with defined config and run it as main process.
    :return: None
    """

    sim = Simulator(**config["Simulator"])
    sim.run()


def run_as_subprocess(joints_sync, ready_sync, done_sync, new_joints_sync, simulate_limbs: bool, simulate_joints: bool, simulate_joint_connections: bool):
    """
    Create Simulator instance with defined config and as subprocess of RGBDto3DPose.
    :param joints_sync: Synchronized joints, mp.Array('f', np.zeros([25 * 3]))
    :param ready_sync: Synchronized is ready flag, mp.Event
    :param done_sync: Synchronized is done flag, mp.Event
    :param new_joints_sync: Synchronized flag if there are new joints, mp.Event
    :param simulate_limbs: If limbs are simulated
    :param simulate_joints: If joints are simulated
    :param simulate_joint_connections: If connections are simulated
    :return: None
    """

    config_sim = config["Simulator"]
    config_sim["playback"] = False
    config_sim["as_subprocess"] = True
    config_sim["joints_sync"] = joints_sync
    config_sim["ready_sync"] = ready_sync
    config_sim["done_sync"] = done_sync
    config_sim["new_joints_sync"] = new_joints_sync
    config_sim["simulate_limbs"] = simulate_limbs
    config_sim["simulate_joints"] = simulate_joints
    config_sim["simulate_joint_connections"] = simulate_joint_connections

    sim = Simulator(**config_sim)
    sim.run()


if __name__ == '__main__':
    run()