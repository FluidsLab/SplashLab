import numpy as np
import glob
import cv2


def error(a, b):
    if a == 0:
        if b == 0:
            return 100000
        return abs(a - b) / b
    return abs(a - b) / a


def read_image_folder(folder_path, file_extension='.tif', start=0, end=None, read_color=False):
    files = glob.glob(folder_path + '/*' + file_extension)
    images = []
    for i in files[start:end]:
        if read_color:
            img = cv2.imread(i)
        else:
            img = cv2.imread(i, 0)
        images.append(img)
    images = np.stack(images, 0)
    return images


def read_mp4(video_file):
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


def find_contours(img, threshold1=100, threshold2=200):
    img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
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


def mouse_tracker(event, x, y, flags, param):
    global mouseX, mouseY, pressX, pressY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX, mouseY = x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pressX, pressY = x, y


mouseX, mouseY, pressX, pressY = -5, -5, -3, -3


def feature_selector(images: np.ndarray, step=1, threshold1=100, threshold2=200) -> np.ndarray:
    global mouseX, mouseY, pressX, pressY
    mouseX, mouseY = -5, -5
    pressX, pressY = -3, -3

    cv2.namedWindow('feature_selector')
    cv2.setMouseCallback('feature_selector', mouse_tracker)
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
            print(len(con))
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


def feature_tracker(images: np.ndarray, selected_contour: np.ndarray, func=None, show_images=True, return_images=False,
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
    return tracked_feature, recorded_images if return_images else tracked_feature


if __name__ == "__main__":
    # test_images = np.load('C:/Users/truma/Documents/Code/ComputerVision_ws/data/bird_impact.npy')
    # # test_images = read_image_folder("C:\\Users\\truma\\Documents\Code\\ai_ws\data\\28679_1_93")
    # print('Images loaded')
    #
    # stuff = feature_selector(test_images)
    # contours, colored = feature_tracker(test_images, stuff, show_images=False, return_images=True)
    # animate_images(colored)

    # test_frames = read_mp4('C:/Users/truma/Downloads/samara_seed.avi')
    # bart_frames = np.load('C:/Users/truma/Documents/Code/ComputerVision_ws/data/bird_impact.npy')[500:1000]
    # contour = feature_selector(bart_frames)
    # track_feature, colored = feature_tracker(bart_frames, contour)
    # animate_images(colored)

    img = cv2.imread("C:/Users/truma/Downloads/curvature.jpeg")
    img = np.array([img])
    contour = feature_selector(img)
