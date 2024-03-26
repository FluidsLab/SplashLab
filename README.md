# SplashLab
Python packages intended to assist with experimental research in the field of fluid dynamics. The tools are divided into
two categories: computer vision and dimensional analysis. The entire package can be installed with:
```
pip install splashlab
```
There are two main modules: computer vision and dimensional analysis. As their names suggest computer vision contacts 
classes and functions to assist with computer vision tasks; it is mostly built on top of OpenCV. Dimensional analysis 
has function and classes intended to assist with tasks such as keeping track of units during calculations and finding 
nondimensional groups.

## Computer Vision
### Reading and Displaying Images
The computer vision module contains several functions and classes that assist mainly with processing image and video 
data. The first example shows how to use `read_image_folder` to convert all images in a directory to a single numpy 
array and  display the images with `animate_images`:
```
import splashlab.computer_vision as vision

images = vision.read_image_folder(folder_path, file_extension='.tif', start=0, end=None, step=1, read_color=False)
vision.animate_images(images, wait_time=10, wait_key=False, BGR=True, close=True)
```

## Dimensional Analysis
Dimensional analysis, as the name suggests, helps with dimensional analysis, like applying Buckingham Pi Theorem.