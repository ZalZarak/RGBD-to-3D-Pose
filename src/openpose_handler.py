import sys
import cv2
import os
from sys import platform
import numpy as np

from src.config import config

op_config = config["OpenPoseHandler"]
# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
openpose_python_path_ubuntu = op_config["openpose_python_path_ubuntu"]
model_folder_ubuntu = op_config["model_folder_ubuntu"]

openpose_python_path_win = op_config["openpose_python_path_win"]
openpose_release_path_win = op_config["openpose_release_path_win"]
openpose_bin_path_win = op_config["openpose_bin_path_win"]
model_folder_win = op_config["model_folder_win"]

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
    def __init__(self, params: dict = None):
        """
        Configures and starts OpenPose-Wrapper.

        :param params: OpenPose parameter. If None, takes parameters defined in config.py.
        Visit https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/demo_advanced.md for full list of parameters.
        """

        if params is None:
            params = op_config["params"]
        params["image_dir"] = None  # With this flag openpose accepts custom input in jpg/png/... format. Image dir is not used
        params["model_folder"] = model_folder

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        self.op_wrapper = op_wrapper

    def push_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, any]:
        """
        Pushes the frame to OpenPose.

        :param frame: a raw color image
        :return: Pixels of joints (np.ndarray([25, 2]), confidences ([np.ndarray([25, 1]), image with drawn joints and connections
        """

        # Transform raw frame to png because OpenPose cannot work with raw images.
        _, encoded_image = cv2.imencode('.png', frame)
        frame = cv2.imdecode(encoded_image, 1)

        # Push png frame to openpose
        datum = op.Datum()
        datum.cvInputData = frame
        self.op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

        if datum.poseKeypoints is not None:
            # return result if success
            return datum.poseKeypoints[0, :, :2], datum.poseKeypoints[0, :, 2], datum.cvOutputData
        else:
            # return default values if no success
            return np.zeros([25, 2]), np.zeros([25, 1]), datum.cvOutputData
