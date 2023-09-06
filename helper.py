import cv2
import numpy as np
from cfonts import render as render_text
import time

from simulator import pairs


def print_countdown(seconds):
    def print_in_big_font(text):
        rendered_text = render_text(text, gradient=['red', 'yellow'], align='center', size=(80, 40))
        print(rendered_text, end='\r')

    if seconds >= 1:
        for i in range(seconds, 0, -1):
            print_in_big_font(f"{i:02d}")
            time.sleep(1)

    print_in_big_font("GO!")
    print()  # Add a new line after the countdown ends


def draw_pixel_grid(image):
    rows, cols = image.shape[:2]
    height, width = image.shape[:2]
    grid_image = image.copy()

    for row in range(0, rows, 50):
        y = int((row / rows) * height)
        cv2.line(grid_image, (0, y), (width, y), (0, 0, 255), 1)
        cv2.putText(grid_image, str(row), (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    for col in range(0, cols, 50):
        x = int((col / cols) * width)
        cv2.line(grid_image, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.putText(grid_image, str(col), (x + 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return grid_image


def show(name: str, frame, joints: np.ndarray = None):
    if joints is not None:
        frame = frame.copy()    # copy because cv2 cannot handle views of ndarrays e.g. if they were rotated etc.
        joints = joints.astype(int)
        for joint in joints:
            if joint.any() != 0:
                pass
                cv2.circle(frame, joint, 3, (255, 255, 255), -1)
        for pair in pairs:
            point1, point2 = joints[pair[0]], joints[pair[1]]
            if point1.any() != 0 and point2.any() != 0:
                cv2.line(frame, point1, point2, (128, 128, 128), 2)

    cv2.imshow(name, draw_pixel_grid(frame))
