import math
from math import sqrt

import numpy as np
import pybullet as p
import pybullet_data


# from openpose_handler import OpenPoseHandler


class Visualizer:
    #TODO: Collision shapes: Dont forget to activate/deactivate
    def __init__(self):
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
            "foot": 0.15
        }

        self.limbs = {}

        # Connect to the PyBullet physics server
        physicsClient = p.connect(p.GUI)

        # Set the camera position and orientation
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-45,
                                     cameraTargetPosition=[0, 0, 0])

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

        self.deactivate_limbs(self.limbs.keys())

        self.limbs_directions = {key: [0, 0, 1] for key in self.limbs}

    def deactivate_limbs(self, names: [str]):
        for name in names:
            p.changeVisualShape(self.limbs[name], -1, rgbaColor=[0, 0, 0, 0])

    def activate_limbs(self, names: [str]):
        for name in names:
            p.changeVisualShape(self.limbs[name], -1, rgbaColor=[0, 0, 0.9, 0.5])

    def move_limbs(self, joints: dict[str, list[np.ndarray]]):
        missing = self.limbs.keys() - joints.keys()
        self.deactivate_limbs(missing)

        for limb, joint in joints.items():
            limb_id = self.limbs[limb]

            if limb in ["head", "handL", "handR"]:
                p.resetBasePositionAndOrientation(limb_id, joint[0], p.getQuaternionFromEuler([0, 0, 0]))
            else:
                coord1, coord2 = joint

                # Calculate midpoint
                midpoint = (coord1 + coord2) / 2

                direction = coord2 - coord1
                current_direction = self.limbs_directions[limb]

                # Calculate the orientation quaternion
                rotation_axis = np.cross(current_direction, direction)
                rotation_angle = np.arccos(np.dot(current_direction, direction) / (np.linalg.norm(current_direction) * np.linalg.norm(direction)))
                orientation = p.getQuaternionFromAxisAngle(rotation_axis, rotation_angle)

                # Update existing cylinder's properties
                p.resetBasePositionAndOrientation(limb_id, midpoint, orientation)

                # Save direction for next time
                self.limbs_directions[limb] = direction

            p.changeVisualShape(limb_id, -1, rgbaColor=[0, 0, 0.9, 0.5])


def visualize_points(point_list, connection_tuples):
    # Create a dictionary to store sphere IDs for each point
    sphere_ids = {}

    # Visualize each point in the list
    for point_id, point in enumerate(point_list):
        if not point[2] == 0:
            x, z, y = point

            # Create a sphere at the point's position
            sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.9, 0, 0, 1.0])

            # Add the sphere to the simulation
            body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[x, y, z])

            # Store the sphere ID in the dictionary
            sphere_ids[point_id] = body_id

    # Visualize connections between points
    for connection in connection_tuples:
        point_id1, point_id2 = connection

        # Retrieve the corresponding sphere IDs
        if point_id1 in sphere_ids and point_id2 in sphere_ids:
            sphere_id1 = sphere_ids[point_id1]
            sphere_id2 = sphere_ids[point_id2]

            # Get the positions of the spheres
            pos1, _ = p.getBasePositionAndOrientation(sphere_id1)
            pos2, _ = p.getBasePositionAndOrientation(sphere_id2)

            # Draw a line between the points
            line_id = p.addUserDebugLine(lineFromXYZ=pos1, lineToXYZ=pos2, lineColorRGB=[0, 0, 0.9], lineWidth=20)

def convert_openpose_coords(coords: np.ndarray) -> dict[str, list[np.ndarray]]:
        ret = {}

        def add_to_ret(name: str, positions: list[int]):
            if all([not np.array_equal(coords[pos], (0, 0, 0)) for pos in positions]):
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

def visualize(joints: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]], point_list, connection_tuples):
    vis = Visualizer()
    vis.move_limbs(joints)
    visualize_points(point_list, connection_tuples)
    while True:
        p.stepSimulation()
    p.disconnect()


if __name__ == '__main__':
    pairs = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15),
             (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
    # Define the person's height and initial position
    person_height = 1.98  # in meters
    initial_z = 1.0  # in meters

    point_list = np.array([
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

    joints = convert_openpose_coords(point_list)

    visualize(joints, point_list, pairs)