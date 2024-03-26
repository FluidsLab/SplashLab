import pandas as pd
import numpy as np
import glob
import cv2

from typing import Any, Callable, Dict
from dataclasses import dataclass, field
from splashlab.dimensional_analysis.util import Util


def error(a, b):
    if a == 0:
        if b == 0:
            return 100000
        return abs(a - b) / b
    return abs(a - b) / a


def read_image_folder(folder_path, file_extension='.tif', start=0, end=None, step=1, read_color=False):
    files = glob.glob(folder_path + '/*' + file_extension)
    images = []
    for i in files[start:end:step]:
        if read_color:
            img = cv2.imread(i)
        else:
            img = cv2.imread(i, 0)
        images.append(img)
    images = np.stack(images, 0)
    return images


def read_video(video_file):
    video = cv2.VideoCapture(video_file)
    success = True
    frames = []
    while success:
        success, frame = video.read()
        if success:
            frames.append(frame)
    frames = np.array(frames)
    return frames


def write_video(output_file, images, framerate=20, color=False, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):
    video = cv2.VideoWriter(output_file, fourcc, framerate, (images.shape[2], images.shape[1]), color)
    for i in images:
        video.write(i)
    video.release()


def animate_images(images, wait_time=10, wait_key=False, BGR=True, close=True):
    window_name = 'image'
    for i, image in enumerate(images):
        cv2.setWindowTitle(window_name, str(i))
        cv2.imshow(window_name, image if BGR or image.shape[-1] != 3 else cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Press Q on keyboard to exit
        if cv2.waitKey(wait_time) & 0xFF == 27:  # ord('q'):
            break
        if wait_key:
            k = cv2.waitKey(0)
            if k == 27:
                break
    if close:
        cv2.destroyAllWindows()


def find_contours(img, threshold1=100, threshold2=200, blur=3):
    img_blur = cv2.GaussianBlur(img, (blur, blur), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


def iou_contour_match(contours, target_contour):
    best_iou = 0
    matching_contour = None
    x1, y1, w1, h1 = cv2.boundingRect(target_contour)
    for cnt in contours:
        x2, y2, w2, h2 = cv2.boundingRect(np.array(cnt))
        intersection_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - intersection_area
        iou = intersection_area / float(max(union_area, 1e-6))
        if iou > best_iou:
            best_iou = iou
            matching_contour = cnt
    return matching_contour


def simple_contour_match(contours, target_contour):
    # TODO Try using two contours combined to improve matching accuracy
    err = 1000000
    for con in contours:
        # shape_err = cv2.matchShapes(target_contour, con, 1, parameter=0)
        length_error = error(con.shape[0], target_contour.shape[0])
        position_error = error(np.mean(target_contour[:, :, 0]), np.mean(con[:, :, 0])) + error(
            np.mean(target_contour[:, :, 1]), np.mean(con[:, :, 1]))
        if length_error + position_error < err:
            err = length_error + position_error
            matching_contour = con
    return matching_contour


def generate_gif(images_location: str, output_file: str) -> None:
    # TODO add logic so it only reads images in the folder
    img_array = []
    for filename in glob.glob(images_location):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('complete')


def track_mouse(event, x, y, flags, param):
    global mouseX, mouseY, pressX, pressY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pressX, pressY = x, y


mouseX, mouseY, pressX, pressY = -5, -5, -3, -3


def select_contour(images: np.ndarray, step=10, threshold1=100, threshold2=200) -> np.ndarray:
    mouse = Mouse()
    if len(images) == 1 or isinstance(images, list):
        images = np.array(images)

    window_name = 'Select Contour'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse.tracker)

    selected = False
    selected_contour = None
    while not selected:
        mouse.index = max(0, mouse.index)
        mouse.index = min(mouse.index, len(images) - 1)
        cv2.setWindowTitle(window_name, str(mouse.index))
        img = images[mouse.index]# if images.shape[-1] != 3 else cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        contours, _ = find_contours(img, threshold1, threshold2)
        img_contour = cv2.drawContours(image=img.copy() if images.shape[-1] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), contours=contours, contourIdx=-1,
                                       color=(100, 200, 255), thickness=1, lineType=cv2.LINE_AA)

        for con in contours:
            dist = abs(cv2.pointPolygonTest(con, (mouse.move.x, mouse.move.y), True))
            if dist < 2:
                img_contour = cv2.drawContours(image=img_contour, contours=con, contourIdx=-1, color=(255, 100, 50),
                                               thickness=2, lineType=cv2.LINE_AA)
            if mouse.pressed and dist < 2:
                selected_contour = con
                selected = True
                break

        cv2.imshow(window_name, cv2.cvtColor(img_contour, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('x'):
            threshold2 += step
        if k == ord('z'):
            threshold2 -= step
        if k == ord('s'):
            threshold1 += step
        if k == ord('a'):
            threshold1 -= step
        elif k == 27:  # 'ESC' key
            break

    cv2.destroyAllWindows()
    return selected_contour, mouse.index, (threshold1, threshold2)


def track_contour(images: np.ndarray, selected_contour: np.ndarray, func=lambda x: x, show_images=True, return_images=False,
                    threshold1=100, threshold2=200):
    tracked_feature = []
    recorded_images = []
    for j, image in enumerate(images):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image if show_images or return_images else image
        contours, _ = find_contours(img, threshold1, threshold2)
        selected_contour = iou_contour_match(contours, selected_contour)

        if func is not None:
            tracked_feature.append(func(selected_contour))

        if show_images or return_images:
            img_contour = cv2.drawContours(image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), contours=contours,
                                           contourIdx=-1, color=(100, 200, 255), thickness=1, lineType=cv2.LINE_AA)
            img_contour = cv2.drawContours(image=img_contour, contours=selected_contour, contourIdx=-1,
                                           color=(255, 125, 75), thickness=3, lineType=cv2.LINE_AA)

            if show_images:
                cv2.setWindowTitle('image', str(j))
                cv2.imshow('image', cv2.cvtColor(img_contour, cv2.COLOR_RGB2BGR))
                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # 'ESC' key
                    break

        if return_images:
            recorded_images.append(img_contour)

    if show_images:
        cv2.destroyAllWindows()
    return (tracked_feature, recorded_images) if return_images else tracked_feature


def track_stable_point(cnts):
    leftmost = []
    rightmost = []
    topmost = []
    bottommost = []
    locations = ['leftmost', 'rightmost', 'topmost', 'bottommost']
    for c in cnts:
        leftmost.append(tuple(c[c[:, :, 0].argmin()][0]))
        rightmost.append(tuple(c[c[:, :, 0].argmax()][0]))
        topmost.append(tuple(c[c[:, :, 1].argmin()][0]))
        bottommost.append(tuple(c[c[:, :, 1].argmax()][0]))
    ma = np.array([leftmost, rightmost, topmost, bottommost])[:, :, 0].max(-1)
    mi = np.array([leftmost, rightmost, topmost, bottommost])[:, :, 0].min(-1)
    index = (ma-mi).argmin()
    return locations[index], np.array([leftmost, rightmost, topmost, bottommost])[index]


@dataclass
class Pixel:
    x: int
    y: int


flag_dict = {
        0: (False, False, False),
        8: (True, False, False),
        16: (False, True, False),
        24: (True, True, False),
        32: (False, False, True),
        40: (True, False, True),
        48: (False, True, True),
        56: (True, True, True)
    }


@dataclass
class Mouse:
    def __init__(self):
        self.down = Pixel(0, 0)
        self.move = Pixel(-10, -10)
        self.up = Pixel(0, 0)
        self.index = 0
        self.pressed: bool = False
        self.shift: bool = False
        self.alt: bool = False
        self.ctrl: bool = False

    def tracker(self, event, x, y, flags, param):
        if flags in flag_dict:
            self.ctrl, self.shift, self.alt = flag_dict[flags]

        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.up = Pixel(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            self.move = Pixel(x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.down = Pixel(x, y)
            self.pressed = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.up = Pixel(x, y)
            self.pressed = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if np.sign(flags) > 0:
                self.index += 1
            elif np.sign(flags) < 0:
                self.index -= 1


def measure_images(images: iter, image_names: iter = None, measure_type: str = '', show_help: bool = False,
                   exit_on_mouse_release: bool = True):
    operations = {'p': 'Pointer', 'c': 'Circle', 'e': 'Ellipse', 'd': 'Distance', 'l': 'Line', 'r': 'Rectangle'}
    operation = measure_type if measure_type else 'p'
    mouse = Mouse()
    shift = False

    window_name = 'measure'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse.tracker)
    while True:
        measurement = {}
        mouse.index = max(0, mouse.index)
        mouse.index = min(mouse.index, len(images) - 1)
        frame_name = (image_names[mouse.index][-1] if isinstance(image_names[mouse.index], list) else image_names[
            mouse.index]) if image_names is not None else mouse.index

        font = cv2.FONT_HERSHEY_SIMPLEX
        fresh_img = cv2.cvtColor(images[mouse.index], cv2.COLOR_GRAY2RGB)
        if show_help:
            fresh_img = cv2.putText(fresh_img, 'Pointer: p, Distance: d, Circle: c, Rectangle: r, Ellipse: e', (10, 30),
                                    font, .8, (200, 100, 0), 1, cv2.LINE_AA)
        x = max(0, min(mouse.move.x, fresh_img.shape[1] - 1)) if mouse.pressed else max(0, min(mouse.up.x,
                                                                                               fresh_img.shape[1] - 1))
        y = max(0, min(mouse.move.y, fresh_img.shape[0] - 1)) if mouse.pressed else max(0, min(mouse.up.y,
                                                                                               fresh_img.shape[0] - 1))
        shift = mouse.shift if mouse.pressed else shift
        fresh_img[:, max(0, min(mouse.move.x, fresh_img.shape[1] - 1))] += 55
        fresh_img[max(0, min(mouse.move.y, fresh_img.shape[0] - 1)), :] += 55

        if mouse.down.x > 0 and mouse.down.y > 0 and y > 0 and y > 0:
            if operation == 'd':
                cv2.line(fresh_img, (mouse.down.x, mouse.down.y), (x, y), (200, 100, 0), 2)
                measurement['distance'] = np.sqrt((mouse.down.x - x) ** 2 + (mouse.down.y - y) ** 2)
                cv2.setWindowTitle(window_name, (
                    f'Image: {frame_name} {operations[operation]} = {measurement["distance"]:.2f} {"pixels"}'))

            elif operation == 'l':
                measurement['start'] = (mouse.down.x, mouse.down.y)
                measurement['end'] = (x, y)
                cv2.line(fresh_img, measurement['start'], measurement['end'], (200, 100, 0), 2)
                cv2.setWindowTitle(window_name, (
                    f'Image: {frame_name}, {operations[operation]}, Start = {measurement["start"]}, End = {measurement["end"]}'))

            elif operation == 'r':
                measurement['top_left'] = (min(mouse.down.x, x), min(mouse.down.y, y))
                measurement['bottom_right'] = (max(mouse.down.x, x), max(mouse.down.y, y))
                cv2.rectangle(fresh_img, measurement['top_left'], measurement['bottom_right'], (200, 100, 0), 2)
                cv2.setWindowTitle(window_name, (
                    f'Image: {frame_name} {operations[operation]}, Top Left: {measurement["top_left"]}, Bottom Right: {measurement["bottom_right"]}'))

            elif operation == 'c':
                measurement['center'] = mouse.down.x, mouse.down.y
                measurement['radius'] = int(np.sqrt((mouse.down.x - x) ** 2 + (mouse.down.y - y) ** 2))

                cv2.circle(fresh_img, measurement['center'], measurement['radius'], (200, 100, 0), 2)
                cv2.setWindowTitle(window_name, (
                    f'Image: {frame_name} {operations[operation]}, Center = {measurement["center"]}, Radius = {measurement["radius"]}'))

            elif operation == 'e':
                if shift:
                    measurement['center'] = mouse.down.x, mouse.down.y
                    measurement['axes'] = (abs(mouse.down.x - x), abs(mouse.down.y - y))
                else:
                    measurement['center'] = int(np.mean([mouse.down.x, x])), int(np.mean([mouse.down.y, y]))
                    measurement['axes'] = (abs(mouse.down.x - x) // 2, abs(mouse.down.y - y) // 2)

                cv2.ellipse(fresh_img, measurement['center'], measurement['axes'], 0, 0, 360, (200, 100, 0), 2)
                cv2.setWindowTitle(window_name, (
                    f'Image: {frame_name} {operations[operation]}, Center = {measurement["center"]}, Axes = {measurement["axes"]}'))

        if operation == 'p' or not (mouse.down.x or mouse.down.y):
            cv2.setWindowTitle(window_name, (
                f'Image: {frame_name}, Operation: {operations[operation]}, Pixel = {(x, y)}, Pixel Values = {fresh_img[y, x, :]}'))

        cv2.imshow(window_name, fresh_img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            if not exit_on_mouse_release:
                break
            elif mouse.down.x > 0 or mouse.down.y > 0:
                mouse.down = Pixel(0, 0)
            else:
                break

        elif not measure_type and chr(k) in operations:
            operation = chr(k)

        if (mouse.down.x <= 0 or mouse.down.y <= 0) and (mouse.up.x > 0 or mouse.up.y > 0):
            mouse.up = Pixel(-10, -10)

        elif exit_on_mouse_release and (mouse.down.x > 0 or mouse.down.y > 0) and (mouse.up.x > 0 or mouse.up.y > 0):
            break

    cv2.destroyAllWindows()
    return measurement


def zoom_img(img, zoom):
    """
    Simple image zooming without boundary checking.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    """

    return cv2.resize(img, (0, 0), fx=zoom, fy=zoom)[
        int(round((img.shape[0] * (zoom - 1)))) // 2: int(round((img.shape[0] * (zoom + 1)))) // 2,
        int(round((img.shape[1] * (zoom - 1)))) // 2: int(round((img.shape[1] * (zoom + 1)))) // 2,
        ]


def define_three_point_circle(p1: Pixel, p2: Pixel, p3: Pixel):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2.x * p2.x + p2.y * p2.y
    bc = (p1.x * p1.x + p1.y * p1.y - temp) / 2
    cd = (temp - p3.x * p3.x - p3.y * p3.y) / 2
    det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)

    if abs(det) < 1.0e-6:
        return None, np.inf

    # Center of circle
    cx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) / det
    cy = ((p1.x - p2.x) * cd - (p2.x - p3.x) * bc) / det

    radius = np.sqrt((cx - p1.x) ** 2 + (cy - p1.y) ** 2)
    return (int(cx), int(cy)), int(radius)


def select_three_point_circle(image, magnification=5, **kwargs):
    mouse = Mouse()
    mouse.index = 5 * magnification
    point1, point2, point3 = None, None, None
    cv2.namedWindow(kwargs.get('window_name', 'three_point_circle'))
    cv2.setMouseCallback(kwargs.get('window_name', 'three_point_circle'), mouse.tracker)
    while True:
        x, y = mouse.move.x, mouse.move.y
        mouse.index = max(0, mouse.index)
        zoom = mouse.index / 5 + 1
        h, w = image.shape
        img = image.copy()

        if not (None in [point1, point2, point3]):
            cv2.circle(img, *define_three_point_circle(point1, point2, point3), 255, 2)
        # if y % 2 == 0 and x % 2 == 0:
        ySlice = slice(max(0, y - 50), min(h, y + 50))
        xSlice = slice(max(0, x - 50), min(w, x + 50))
        img[ySlice, xSlice] = zoom_img(img[ySlice, xSlice], zoom)

        cv2.imshow(kwargs.get('window_name', 'three_point_circle'), img)

        if mouse.pressed:
            if point1 is None:
                point1 = mouse.down
            elif point2 is None:
                point2 = mouse.down
            elif point3 is None:
                point3 = mouse.down
            mouse.pressed = False
        k = cv2.waitKey(1) & 0xFF

        if k in [27, 13, 32]:  # ESC is 27, ENTER is 13, SPACE is 32
            break
        elif chr(k) == 'z':
            if point3 is not None:
                point3 = None
            elif point2 is not None:
                point2 = None
            elif point1 is not None:
                point1 = None
    cv2.destroyAllWindows()
    return point1, point2, point3


if __name__ == "__main__":
    impact_frame = 113
    images = read_image_folder(r'E:\ALAYESH_2023_2DSPLASH\data\30_47_cal001', read_color=True, start=impact_frame-5, end=impact_frame)
    cnt, frame_num, (threshold1, threshold2) = select_contour(images)
    _, imgs = track_contour(images, cnt, show_images=False, return_images=True, threshold1=threshold1, threshold2=threshold2)
    animate_images(imgs, wait_key=True)
