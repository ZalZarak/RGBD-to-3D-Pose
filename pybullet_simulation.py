import pybullet as p
import time


def create_sphere(position):
    sphere_id = p.createVisualShape(p.GEOM_SPHERE, radius=1)
    p.createMultiBody(baseVisualShapeIndex=sphere_id, basePosition=position)
    return sphere_id


def update_sphere_positions(spheres, coordinates):
    for sphere, coord in zip(spheres, coordinates):
        p.resetBasePositionAndOrientation(sphere, coord, [0, 0, 0, 1])


def update_simulation(coordinates):
    # Update the positions of the spheres
    update_sphere_positions(spheres, coordinates)

    # Step the simulation
    p.stepSimulation()


def main(resolution: tuple[int, int]):
    # Connect to PyBullet and set up the simulation
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)

    # Create spheres at the initial coordinates
    global spheres
    spheres = []
    coordinates = []
    for _ in range(resolution[0] * resolution[1]):
        spheres.append(create_sphere([-1, -1, -1]))
        coordinates.append([-1, -1, -1])

    try:
        while True:
            # Example: Generate new coordinates (you should replace this with your data source)
            coordinates = [(x + 0.01, y + 0.01, z) for x, y, z in coordinates]

            # Call the function to update the simulation with the new coordinates
            update_simulation(coordinates)

            # Pause for a short time (adjust as needed to control the update rate)
            time.sleep(0.1)
    finally:
        # Close the PyBullet connection when the script exits
        p.disconnect()