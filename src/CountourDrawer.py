from typing import List, Optional
from CustomTypes import Image, Contour
import matplotlib.pyplot as plt
import cv2 as cv



def add_contours_to_frame(frame: Image):
    """
    Apply's Contours around Shapes to a given Frame

    Args:
        frame: Image (Alias for np.ndarray) (Expects image in BGR color)

    Returns:
        processed_frame: Image(Alias for np.ndarray)
    """
    import time

    
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    adaptiveThresholdImage = apply_adaptive_thresholding(grayscale)
    
    SansNoise = eliminate_noise(adaptiveThresholdImage)
    
    contours = calculate_countours(SansNoise)

    centroids = calculate_centroids(contours)

    draw_centroids(centroids, frame)
    ImageWithContours = cv.drawContours(frame, contours, -1, (255, 255, 0), 3)
    
    return ImageWithContours

def apply_adaptive_thresholding(frame: Image) -> Image:
    """
    For every Pixel, The mean is calculated over a block of odd size. 
    The mean is then subtracted by the constant C. If the pixel intensity is lower 
    than the mean - C is set to 0, otherwise it is set to 255. 
    """

    Constant = 2
    BlockSize = 9

    adaptive_gaussian = (
        cv.adaptiveThreshold(frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv.THRESH_BINARY, BlockSize, Constant)
    )

    return adaptive_gaussian    
    pass

def eliminate_noise(frame: Image):
    """
    This applies a Gaussian Blur to the Image and only keeps the
    Very Bright Areas ig
    """
    blur = cv.blur(frame, (9, 9))
    ret, thresh = cv.threshold(blur, 205, 255, cv.THRESH_BINARY) 
    
    return thresh

def calculate_countours(frame: Image):
    contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def calculate_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv.moments(contour)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except ZeroDivisionError:
            continue
        coord = (cx, cy)
        centroids.append(coord)

    return centroids

def draw_centroids(centroids, frame):
    WHITE = (255,255,255)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_thickness = 1
    radius = 5
    for center in centroids:
        cx, cy = center
        cv.putText(frame, f'center {cx, cy}',(cx + 40 ,cy), font, font_size, WHITE, font_thickness,cv.LINE_AA)
        cv.circle(frame, center, radius, WHITE, -1)
    

def plot_images(images: List[Image], titles: Optional[List[str]] = None):
    """
    Plot a list of Black and White images with optional titles.

    Args:
        images: List of images to plot.
        titles: List of titles for each image.
    """
    if not titles:
        titles = [f"Image {i}" for i in range(len(images))]

    cols = min(len(images), 4)
    rows = (len(images) + cols - 1) // cols
    for i in range(len(images)):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()