import time

import cv2
import numpy as np
import pyrealsense2 as rs
from cfonts import render as render_text
import pybullet_simulation as sim
from old.main import visualize_points

frame = 10
count = 0
intrinsics = 0

def estimate_pose():
    pass

def print_countdown(seconds):
    def print_in_big_font(text):
        rendered_text = render_text(text, gradient=['red', 'yellow'], align='center', size=(80, 40))
        print(rendered_text, end='\r')

    for i in range(seconds, 0, -1):
        print_in_big_font(f"{i:02d}")
        time.sleep(1)

    print_in_big_font("GO!")
    print()  # Add a new line after the countdown ends


def process_frame(pipeline: rs.pipeline, colorizer: rs.colorizer, rotate: int):
    # Wait for the next set of frames from the camera
    frames = pipeline.wait_for_frames()

    # Get depth frame
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)

    # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    depth_color_image = np.rot90(depth_color_image, k=rotate)
    color_image = np.rot90(color_image, k=rotate)

    # Render image in opencv window
    cv2.imshow("Depth Stream", depth_color_image)
    cv2.imshow("Color Stream", color_image)
    """global count
    if count == frame:
        coordinates = []
        coordinates2 = np.zeros([480, 270, 3])
        #depths = np.zeros([480, 270])
        for x in range(479-449, 479-204+1, 4):#range(240, 480, 2):
            for y in range(42, 186+1, 4): #range(54, 150, 2):
                depth = depth_frame.as_depth_frame().get_distance(x, y)
                #depths[479 - x, y] = depth
                if 0.0 < depth < 3.0:
                    coord = rs.rs2_deproject_pixel_to_point(intrin=intrinsics, pixel=(x,y), depth=depth)
                    coord = [coord[1], coord[0], coord[2]]
                    coordinates.append(coord)
                    coordinates2[479-x, y] = coord
        coord_y_min = np.min(coordinates2[:,:,1])
        for c in coordinates:
            c[1] -= coord_y_min
        visualize_points(coordinates)
    count += 1"""


def prepare_pipeline(resolution: tuple[int, int], fps: int, play_from_file: bool, save_to_file: bool, filename: str = None, countdown: int = None) -> rs.pipeline:
    assert not (play_from_file and save_to_file), "You can't play from file and save to file at the same time"
    if play_from_file or save_to_file:
        assert filename is not None, "Provide file"

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)

    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    if play_from_file:
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, file_name=filename)

    # Enable both depth and color streams
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, fps)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)

    if save_to_file:
        config.enable_record_to_file(filename)
        # Start the pipeline
        pipeline_profile = pipeline.start(config)
        device = pipeline_profile.get_device()
        recorder = device.as_recorder()
        rs.recorder.pause(recorder)
        print_countdown(countdown)
        rs.recorder.resume(recorder)
    else:
        pipeline_profile = pipeline.start(config)

    global intrinsics
    intrinsics = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()



    return pipeline


def stream(duration: float = float("inf"), resolution: tuple[int, int] = (640, 480), fps: int = 30, record_filename: str = None, countdown: int = 3, rotate=1):

    try:
        max_frames = int(duration * fps)    # frames = seconds * fps
    except OverflowError:
        max_frames = duration
    frame_counter = 0
    key = None

    colorizer = rs.colorizer()  # Create colorizer object
    pipeline = prepare_pipeline(resolution, fps, False, record_filename is not None, record_filename, countdown)

    try:
        while key != 27 and frame_counter < max_frames:
            process_frame(pipeline, colorizer, rotate)

            key = cv2.waitKey(1)
            frame_counter += 1

    finally:
        # Stop the pipeline and close the bag file
        cv2.destroyAllWindows()
        pipeline.stop()


def play(file_name: str, resolution: tuple[int, int] = (640, 480), fps: int = 30, rotate=1):

    key = None

    colorizer = rs.colorizer()  # Create colorizer object
    pipeline = prepare_pipeline(resolution, fps, True, False, file_name)

    try:
        # Streaming loop
        while key != 27:
            process_frame(pipeline, colorizer, rotate)
            key = cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()


# stream(resolution=(480, 270), countdown=5, record_filename="test.bag")

play("test.bag", resolution=(480, 270))

