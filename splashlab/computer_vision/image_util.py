import numpy as np
import glob
import cv2

from dataclasses import dataclass


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
            cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()


def find_contours(img, threshold1=100, threshold2=200, blur=3):
    img_blur = cv2.GaussianBlur(img, (blur, blur), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


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


def select_contour(images: np.ndarray, step=1, threshold1=100, threshold2=200) -> np.ndarray:
    global mouseX, mouseY, pressX, pressY
    mouseX, mouseY = -5, -5
    pressX, pressY = -3, -3

    if len(images) == 1 or isinstance(images, list):
        images = np.array(images)

    cv2.namedWindow('feature_selector')
    cv2.setMouseCallback('feature_selector', track_mouse)
    i = 0
    selected = False
    selected_contour = None
    while i < len(images) and not selected:
        cv2.setWindowTitle('feature_selector', str(i))
        img = images[i] if images.shape[-1] !=3 else cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        contours, _ = find_contours(img, threshold1, threshold2)
        img_contour = cv2.drawContours(image=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), contours=contours, contourIdx=-1,
                                       color=(100, 200, 255), thickness=1, lineType=cv2.LINE_AA)

        for con in contours:
            dist = abs(cv2.pointPolygonTest(con, (mouseX, mouseY), True))
            if dist < 2:
                img_contour = cv2.drawContours(image=img_contour, contours=con, contourIdx=-1, color=(255, 100, 50),
                                               thickness=2, lineType=cv2.LINE_AA)
            if mouseX == pressX and mouseY == pressY and dist < 2:
                selected_contour = con
                selected = True
                break

        cv2.imshow('feature_selector', cv2.cvtColor(img_contour, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            i += step
        elif k == 27:  # 'ESC' key
            break

    cv2.destroyAllWindows()
    return selected_contour


def track_contour(images: np.ndarray, selected_contour: np.ndarray, func=lambda x: x, show_images=True, return_images=False,
                    threshold1=100, threshold2=200):
    tracked_feature = []
    recorded_images = []
    for j, image in enumerate(images):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image if show_images or return_images else image
        contours, _ = find_contours(img, threshold1, threshold2)
        selected_contour = simple_contour_match(contours, selected_contour)

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


@dataclass
class Pixel:
    x: int
    y: int


@dataclass
class Mouse:
    down: Pixel = Pixel(0, 0)
    move: Pixel = Pixel(-10, -10)
    up: Pixel = Pixel(0, 0)
    index = 0
    pressed: bool = False
    shift: bool = False
    alt: bool = False
    shift: bool = False
    ctrl: bool = False
    flag_dict = {0: (False, False, False), 8: (True, False, False), 16: (False, True, False), 24: (True, True, False),
                 32: (False, False, True), 40: (True, False, True), 48: (False, True, True), 56: (True, True, True)}

    def tracker(self, event, x, y, flags, param):
        if flags in self.flag_dict:
            self.ctrl, self.shift, self.ctrl = self.flag_dict[flags]

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


if __name__ == "__main__":
    # test_images = np.load('C:/Users/truma/Documents/Code/ComputerVision_ws/data/bird_impact.npy')
    # test_images = read_image_folder(r"C:\Users\truma\Documents\Code\ai_ws\data\28679_1_93")
    # test_images = read_image_folder(r'C:\Users\truma\OneDrive\Desktop\Test Folder', file_extension='.jpeg')
    test_images = read_image_folder(r"C:\Users\truma\Documents\MATLAB\28679_1_89", step=1000)
    print(test_images.shape)
    print('Images loaded')
    #
    # stuff = feature_selector(test_images)
    # contours, colored = feature_tracker(test_images, stuff, show_images=False, return_images=True)
    # animate_images(test_images)

    # test_frames = read_mp4('C:/Users/truma/Downloads/samara_seed.avi')
    # bart_frames = np.load('C:/Users/truma/Documents/Code/ComputerVision_ws/data/bird_impact.npy')[500:1000]
    # contour = feature_selector(bart_frames)
    # track_feature, colored = feature_tracker(bart_frames, contour)
    # animate_images(colored)

    # img = cv2.imread("C:/Users/truma/Downloads/curvature.jpeg", 0)
    # contour = feature_selector([img])
    # print(len(contour))
