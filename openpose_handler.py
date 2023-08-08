import random
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_python_path = "/home/pk/programs/openpose/build/python"
model_folder = "/home/pk/programs/openpose/models/"

try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append(dir_path + '/../../python/openpose/Release')
        sys.path.append(openpose_python_path)
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('/home/pk/programs/openpose/build/python');
        sys.path.append(openpose_python_path)
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenPoseHandler:
    op_wrapper: any

    def __init__(self):
        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", default="imgs/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        parser.add_argument("--hand", default=True, help="Enable to disable the visual display.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        # params["model_folder"] = "/home/pk/programs/openpose/models/"
        params["model_folder"] = model_folder
        # params["hand"] = True

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        self.op_wrapper = op_wrapper

    def push_frame(self, frame: np.ndarray, show_video: bool) -> np.ndarray:
        _, encoded_image = cv2.imencode('.png', frame)
        frame = cv2.imdecode(encoded_image, 1)

        datum = op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))
        if show_video:
            cv2.imshow("Joint-Stream", datum.cvOutputData)

        return datum.poseKeypoints
