# SplashLab
Python packages intended to assist with experimental research in the field of fluid dynamics. The tools are divided into two categories: computer vision and dimensional analysis. The entire package can be installed with `pip install splashlab`.

### Computer Vision
The computer vision module contains several functions and classes that assist mainly with processing image and video data.
The first example shows how to use `read_image_folder` to convert all images in a directory to a single numpy array and 
display the images with `animate_images`:
```
import splashlab.computer_vision as vision

images = read_image_folder(folder_path, file_extension='.tif', start=0, end=None, step=1, read_color=False)
animate_images(images, wait_time=10, wait_key=False, BGR=True, close=True)
```

### Dimensional Analysis
