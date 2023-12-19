from src import debug, rgbd_to_3d_pose, simulator
from src.config import config

if __name__ == '__main__':
    if config["run_from"] == 0:
        rgbd_to_3d_pose.run()
    elif config["run_from"] == 1:
        simulator.run()
    elif config["run_from"] == 2:
        if config["Debug"]["mode"] == 0:
            debug.debug_color_mask()
        elif config["Debug"]["mode"] == 1:
            debug.debug_search_area()
        elif config["Debug"]["mode"] == 2:
            debug.debug_length(**config["Debug"]["length_args"])
        elif config["Debug"]["mode"] == 3:
            debug.view_coordinates(**config["Debug"]["view_coordinates_args"])
    else:
        raise ValueError("run_from: 0 or 1, debug mode: 0-3")
