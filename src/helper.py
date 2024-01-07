import cv2
import numpy as np
from cfonts import render as render_text
import time


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


def show(name: str, frame, joints: np.ndarray = None, pairs: list[tuple[int, int]] = None):
    if joints is not None:
        frame = frame.copy()    # copy because cv2 cannot handle views of ndarrays e.g. if they were rotated etc.
        joints = joints.astype(int)
        for joint in joints:
            if joint.any() != 0:
                cv2.circle(frame, joint, 3, (255, 255, 255), -1)
        for pair in pairs:
            point1, point2 = joints[pair[0]], joints[pair[1]]
            if point1.any() != 0 and point2.any() != 0:
                cv2.line(frame, point1, point2, (128, 128, 128), 2)

    cv2.imshow(name, draw_pixel_grid(frame))


def show_mask(name: str, frame, color_range: np.ndarray):
    mask = (np.array(cv2.inRange(frame, color_range[0], color_range[1])) == 0)

    mask_image = np.copy(frame)
    mask_image[mask] = 0

    cv2.imshow(name, mask_image)


def generate_base_search_area(deviation: int, skip: int) -> list[tuple[int, int]]:
    if deviation <= 0 or skip < 0:
        return []
    search = []
    skip += 1
    deviation = deviation - deviation % skip
    for i in range(-deviation, deviation + 1, skip):
        for j in range(-deviation, deviation + 1, skip):
            search.append((i, j))
    search.sort(key=lambda a: a[0] ** 2 + a[1] ** 2)
    # search.pop(0) commented out for validation via color.
    return search


def generate_search_pixels(pixel: tuple[int, int], joint_id: int, base_search_area: dict[int, list[tuple[int, int]]], resolution: tuple[int, int]):
    search = map(lambda a: (a[0] + pixel[0], a[1] + pixel[1]), base_search_area[joint_id])
    search = filter(lambda a: 0 <= a[0] < resolution[0] and 0 <= a[1] < resolution[1], search)
    return search


class NoFilter:
    def process(self, a):
        return a


class NoList:
    def append(self, a):
        return
