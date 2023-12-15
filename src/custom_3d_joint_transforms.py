import numpy as np

# define your own transforms for extracted 3d joints.
# functions must take and return np.ndarray([25, 3]).
# This will be applied after 3d joints are extracted and validated.
# But before any other transforms, e.g. the joints are in camera space, not in "real" space.
# So, neither flips (90° camera rotations), nor translations nor rotation were applied at this point.
# undetected joints are (0,0,0).
# unvalidated joints have z=0, e.g. (x,y,0).
# This should stay this way, e.g. (0,0,0) should stay (0,0,0) and (x,y,0) should stay (x,y,0).
# A joint with (x,y,z) with z!=0 must not become (a,b,0).


def estimate_nose_from_head_joints(joints_3d: np.ndarray) -> np.ndarray:
    """
    If nose was not detected, try estimating it from validated eye/ear joints.

    :param joints_3d: Human joints.
    :return: Human joints with estimated nose coordinates if necessary
    """

    # reduce head to nose
    if all(joints_3d[0] == 0):  # nose not detected
        if joints_3d[15, 2] != 0 and joints_3d[16, 2] != 0:  # both eyes validated
            joints_3d[0] = (joints_3d[15] + joints_3d[16]) / 2
        elif joints_3d[15, 2] != 0:  # one eye validated
            joints_3d[0] = joints_3d[15]
        elif joints_3d[16, 2] != 0:  # one eye validated
            joints_3d[0] = joints_3d[16]
        elif joints_3d[17, 2] != 0 and joints_3d[18, 2] != 0:  # both ears validated
            joints_3d[0] = (joints_3d[17] + joints_3d[18]) / 2
        elif joints_3d[17, 2] != 0:  # one ear validated
            joints_3d[0] = joints_3d[17]
        elif joints_3d[18, 2] != 0:  # one ear validated
            joints_3d[0] = joints_3d[18]
    elif joints_3d[0, 2] == 0:  # nose not validated
        if joints_3d[15, 2] != 0 and joints_3d[16, 2] != 0:  # both eyes validated
            joints_3d[0, 2] = (joints_3d[15, 2] + joints_3d[16, 2]) / 2
        elif joints_3d[15, 2] != 0:  # one eye validated
            joints_3d[0, 2] = joints_3d[15, 2]
        elif joints_3d[16, 2] != 0:  # one eye validated
            joints_3d[0, 2] = joints_3d[16, 2]
        elif joints_3d[17, 2] != 0 and joints_3d[18, 2] != 0:  # both ears validated
            joints_3d[0, 2] = (joints_3d[17, 2] + joints_3d[18, 2]) / 2
        elif joints_3d[17, 2] != 0:  # one ear validated
            joints_3d[0, 2] = joints_3d[17, 2]
        elif joints_3d[18, 2] != 0:  # one ear validated
            joints_3d[0, 2] = joints_3d[18, 2]

    return joints_3d
