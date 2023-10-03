from config import config
import rgbd_to_3d_pose
import simulator

if __name__ == '__main__':
    if config["run_from"] == 0:
        rgbd_to_3d_pose.run()
    elif config["run_from"] == 1:
        simulator.run()
    else:
        raise ValueError("run_from: 0 or 1")
