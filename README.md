# Shape Detection Algorithm for Aerial Robotics
*Computer Vision Pipeline for Autonomous Aircraft Navigation*

*Developed for Penn Aerial Robotics Software Challenge - Computer Vision for Autonomous Aircraft Navigation*

## Project Overview

This project implements a robust computer vision algorithm designed for Penn Aerial Robotics' Software Challenge. The algorithm detects and tracks solid shapes on various backgrounds and  from 2D pixel coordinates to 3D world coordinates for real-world applications.

**Key Technologies:** OpenCV, Python, Computer Vision, Pixel -> 3D Coordinate Transformation, Multiprocessing

## How to view results.

Due to Github upload limits, results have been posted in this google drive folder [link](https://drive.google.com/drive/folders/1UlPNVFgMqDX1bOVcVGn67kqH7imDV7Fb?usp=drive_link)
 

## Installation & Usage

### Prerequisites
```bash
pip install opencv-python numpy matplotlib
```

### Running the Algorithm

#### Sequential Processing
```bash
python3 main.py
```

#### Parallel Processing (Recommended)
```bash
python3 MultiProcessing.py
```

---

##  Features

###  Multi-Phase Implementation
- **Phase 1:** Static image shape detection with contour tracing
- **Phase 2:** Real-time video processing with frame-by-frame analysis  
- **Phase 3:** Background-agnostic detection across various environments
- **Phase 4:** 3D coordinate transformation using camera intrinsics



## Technical Implementation

### Core Algorithm Pipeline

```
Input Frame → Grayscale → Adaptive Thresholding → Noise Reduction → Contour Detection → Centroid Calculation → 3D Transformation
```

#### 1. **Adaptive Gaussian Thresholding**
Converts Grayscale images into binary images highlighting each shape using local neighborhood analysis, making solid shapes bright white.


**Image After Thresholding**


<img width="370" height="239" alt="Screenshot 2025-08-30 at 7 15 49 PM" src="https://github.com/user-attachments/assets/4ce41b7a-b724-4e8d-b0e4-f8e7c0b56286" />

#### 2. **Iterative Noise Reduction**
- **Gaussian Blur:** Smooths noisy areas while preserving large shape blocks
- **Binary Thresholding:** Creates clean white shapes on black background
- **Morphological Dilation:** Restores shape edges lost during blur operations

**Image After Noise Reduction**

<img width="370" height="304" alt="Screenshot 2025-08-30 at 7 59 10 PM" src="https://github.com/user-attachments/assets/8aad7ef8-f8ca-4c72-9ecd-d0ce337bf184" />

#### 3. **Contour Detection & Analysis**
Utilizes OpenCV's contour detection to identify shape boundaries and calculate precise centroids in 2D pixel coordinates.

#### 4. **3D Coordinate Transformation**

### Mathematical Foundation

The algorithm extends from 2D pixel detection to 3D world coordinates using the **pinhole camera model**. This transformation converts pixel locations to real-world measurements in inches, enabling spatial positioning.

### Camera Intrinsic Matrix

The camera's intrinsic matrix encodes the fundamental properties of how the camera lens projects 3D world points onto the 2D image plane.

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 
0 & f_y & c_y \\ 
0 & 0 & 1 
\end{bmatrix} = \begin{bmatrix} 2564.32 & 0 & 0 \\ 
0 & 2569.70 & 0 \\ 
0 & 0 & 1 \end{bmatrix}$$

Where `fx` and `fy` are the focal lengths in pixel units, and `cx`, `cy` represent the principal point (image center). The intrinsic matrix establishes the mathematical relationship described by the pinhole projection model: `s[u, v, 1]ᵀ = K[Xc, Yc, Zc]ᵀ`, (u, v are the pixel locations) which tells us how any 3D point gets mapped to a 2D pixel location.

### Calibration and Implementation

Our algorithm leverages a known reference - the circle with 20-inch diameter that appears as ~192 pixels - to establish the conversion factor of 9.6 pixels per inch. With this calibration and the intrinsic matrix, we can reverse the projection process: given a 2D pixel location and the object's size, we calculate where that object exists in 3D space.

```python
def get_3d_coordinates(cx, cy, radius_pixels):
    """
    Calculates the 3D Coordinates of the Centroid
    """
    PIXELS_PER_INCH = 9.6
    intrinsic_matrix = np.array(
        [[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]]
    )
    fx = intrinsic_matrix[0, 0]          # Focal length X
    fy = intrinsic_matrix[1, 1]          # Focal length Y  
    image_center_x = intrinsic_matrix[0, 2]  # Principal point X (0)
    image_center_y = intrinsic_matrix[1, 2]  # Principal point Y (0)

    # Step 1: Convert pixel radius to real-world inches
    radius_inches = radius_pixels / PIXELS_PER_INCH
    
    # Step 2: Calculate depth 
    Zc = radius_inches * fx / radius_pixels
    
    # Step 3: Convert 2D pixel coordinates to 3D world coordinates
    Xc = (cx - image_center_x) * Zc / fx
    Yc = (cy - image_center_y) * Zc / fy

    return float(Xc), float(Yc), float(Zc)
```

### Coordinate System Definition

The output coordinates represent real-world positions relative to the camera:

- **Xc**: Horizontal distance from camera center (inches)

- **Yc**: Vertical distance from camera center (inches)  

- **Zc**: Forward distance from camera (inches)


### Example Output

For a detected circle at pixel position `(cx=640, cy=480)` with `radius_pixels=96`:

```python
Xc, Yc, Zc = get_3d_coordinates(640, 480, 96)
# Result: Xc=25.6", Yc=18.7", Zc=240.0"
# Interpretation: Circle is 25.6" right of center, 18.7" below center, and 240" in front of camera
```
---

## Results & Performance

### Background Agnostic Performance
Successfully detects shapes across:
-  Grass backgrounds (primary use case)
-  Complex textured environments  
-  Varying lighting conditions
-  Multiple shape geometries (circles, squares, triangles)

### Processing Efficiency
- **Sequential Processing:** ~5 minutes for 3 videos
- **Parallel Processing:** ~1.5 minutes for 3 videos (**70% improvement**)
- **CPU Utilization:** Efficient 20% usage on Apple M3 Pro (11-core)

---



