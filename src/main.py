from src import debug, perceptor, simulator
from src.config import config

if __name__ == '__main__':
    if config["run"] == 0:
        perceptor.run()
    elif config["run"] == 1:
        simulator.run()
    elif config["run"] == 2:
        if config["Debug"]["mode"] == 0:
            debug.debug_color_mask()
        elif config["Debug"]["mode"] == 1:
            debug.debug_search_area()
        elif config["Debug"]["mode"] == 2:
            debug.debug_length(**config["Debug"]["length_args"])
        elif config["Debug"]["mode"] == 3:
            debug.view_coordinates(**config["Debug"]["view_coordinates_args"])
    else:
        raise ValueError("run: 0 or 1, debug mode: 0-3")
