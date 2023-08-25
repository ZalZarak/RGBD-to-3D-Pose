
#from listener import listen, pcl
import pybullet as p


def visualize_points(point_list, connection_tuples, point_list2):
    # Connect to the PyBullet physics server
    physicsClient = p.connect(p.GUI)

    # Set the camera position and orientation
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-45,
                                 cameraTargetPosition=[0, 0, 0])

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

    for point_id, point in enumerate(point_list2):
        if not point[2] == 0 and point_list[point_id, 2] == 0:
            x, z, y = point

            # Create a sphere at the point's position
            sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 0.9, 1.0])

            # Add the sphere to the simulation
            body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[x, y, z])

            # Store the sphere ID in the dictionary
            # sphere_ids[point_id] = body_id

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

    # Keep the simulation running until the user closes the window
    while True:
        p.stepSimulation()

    # Disconnect from the physics server
    p.disconnect()


if __name__ == '__main__':
    visualize_points([[2,1,0]])