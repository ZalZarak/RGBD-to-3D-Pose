import random
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

from helper import draw_pixel_grid

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_python_path_ubuntu = "/home/pk/programs/openpose/build/python"
model_folder_ubuntu = "/home/pk/programs/openpose/models/"

openpose_python_path_win = "C:/Users/PK/Documents/Uni/Semester 8/BA/openpose/build2/python/openpose/Release/"
openpose_release_path_win = "C:/Users/PK/Documents/Uni/Semester 8/BA/openpose/build2/x64/Release"
openpose_bin_path_win = "C:/Users/PK/Documents/Uni/Semester 8/BA/openpose/build2/bin"
model_folder_win = "C:/Users/PK/Documents/Uni/Semester 8/BA/openpose/models"

try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append(dir_path + '/../../python/openpose/Release')
        sys.path.append(openpose_python_path_win)
        os.environ['PATH'] = os.environ['PATH'] + ';' + openpose_release_path_win + ';' + openpose_bin_path_win + ';'
        model_folder = model_folder_win
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('/home/pk/programs/openpose/build/python');
        sys.path.append(openpose_python_path_ubuntu)
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        model_folder = model_folder_ubuntu
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenPoseHandler:
    op_wrapper: any
    poseModel = op.PoseModel.BODY_25
    mapping = op.getPoseBodyPartMapping(poseModel)
    reverse_mapping = {v: k for k, v in op.getPoseBodyPartMapping(poseModel).items()}
    #pairs = [(op.getPosePartPairs(op.PoseModel.BODY_25)[i], op.getPosePartPairs(op.PoseModel.BODY_25)[i + 1]) for i in range(0, len(op.getPosePartPairs(op.PoseModel.BODY_25)), 2)]
    #pairs_hr = [f"{op.getPoseBodyPartMapping(op.PoseModel.BODY_25)[a]}-{op.getPoseBodyPartMapping(op.PoseModel.BODY_25)[b]}" for a, b in pairs]
    pairs = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]


    def __init__(self):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        # all parameters defined here: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/demo_advanced.md
        params = {
            "image_dir": None,   # With this flag openpose accepts custom input in jpg/png/... format. Image dir is not used
            "model_folder": model_folder,
            # "hand": True,
            "process_real_time": True,
            "number_people_max": 1,
            # "net_resolution": "-1x736"    # Multiples of 16. If it is increased, the accuracy potentially increases. If it is decreased, the speed increases. For maximum speed-accuracy balance, it should keep the closest aspect ratio possible to the images or videos to be processed. Using -1 in any of the dimensions, OP will choose the optimal aspect ratio depending on the user's input value.
            # "hand_net_resolution": "240x240",
            "render_pose": 1
        }

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        self.op_wrapper = op_wrapper

    def push_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, any]:
        _, encoded_image = cv2.imencode('.png', frame)
        frame = cv2.imdecode(encoded_image, 1)

        datum = op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is not None:
            return datum.poseKeypoints[0, :, :2], datum.poseKeypoints[0, :, 2], datum.cvOutputData
        else:
            return np.zeros([25, 2]), np.zeros([25, 1]), datum.cvOutputData


if __name__ == '__main__':
    print(OpenPoseHandler.mapping)
    print(OpenPoseHandler.pairs)
    print(OpenPoseHandler.pairs_hr)
