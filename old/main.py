
#from listener import listen, pcl
import pybullet as p


def visualize_points(point_list):
    # Connect to the PyBullet physics server
    physicsClient = p.connect(p.GUI)

    # Set the camera position and orientation
    p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=45, cameraPitch=-45,
                                 cameraTargetPosition=[0, 0, 0])

    # Add a large ground plane with the desired color
    """ground_color = [0.8, 0.8, 0.8]  # RGB color in range [0, 1]
    plane_id = p.createCollisionShape(p.GEOM_PLANE)
    ground_visual = p.createVisualShape(p.GEOM_PLANE, rgbaColor=ground_color)
    p.createMultiBody(0, plane_id, -1, [0, 0, 0], [0, 0, 0, 1], ground_visual)"""

    # Visualize each point in the list
    for point in point_list:
        x, z, y = point

        # Create a sphere at the point's position
        sphere_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.9, 0, 0, 1.0])

        # Add the sphere to the simulation
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere_id, baseCollisionShapeIndex=-1, basePosition=[x, y, z])

    # Keep the simulation running until the user closes the window
    while True:
        p.stepSimulation()

    # Disconnect from the physics server
    p.disconnect()

if __name__ == '__main__':
    visualize_points([[2,1,0]])