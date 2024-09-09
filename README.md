# CudaVB

A simple image processing library written in CUDA and target for VB.net.

Each algorithm is implemented in both parallel (CUDA) and non-parallel version. Some of the CUDA algorithms are using [NVIDIA Performance Primitives](https://developer.nvidia.com/npp).

# Implemented Algorithms

- Color transformation
  - RGB to HSV
  - RGB to gray scale
  - Binarize
- Transformation
  - Rotation
  - Bilinear scaling
  - Gaussian pyramid
- Histogram equalization
- Otsu's method
- 2D convolution
- Canny edge detector
- Median filter
- Morphology
  - Dilation
  - Erosion
  - Thinning
- Connected component labeling
- Template matching
- Hough line transform

# References

- [OpenCV](https://github.com/opencv/opencv)
- [NVIDIA Performance Primitives](https://developer.nvidia.com/npp)
- [YACCLAB](https://github.com/prittt/YACCLAB)
- [dustynv/jetson-utils](https://github.com/dusty-nv/jetson-utils)
