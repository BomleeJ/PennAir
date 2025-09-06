from typing import List, Optional
from src.CustomTypes import Image, Contour
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def add_contours_to_frame(frame: Image, Video=True):
    """
    Apply's Contours around Shapes to a given Frame

    Args:
        frame: Image (Alias for np.ndarray) (Expects image in BGR color)

    Returns:
        processed_frame: Image(Alias for np.ndarray)
    """

    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    adaptiveThresholdImage = apply_adaptive_thresholding(grayscale)

    SansNoise = eliminate_noise(adaptiveThresholdImage, Video)

    contours = calculate_countours(SansNoise, Video)

    centroids = calculate_centroids(contours)

    draw_centroids(centroids, frame)
    ImageWithContours = cv.drawContours(frame, contours, -1, (255, 255, 0), 3)

    images = [grayscale, adaptiveThresholdImage, SansNoise, ImageWithContours]
    return ImageWithContours


def apply_adaptive_thresholding(frame: Image) -> Image:
    """
    For every Pixel, The mean is calculated over a block of odd size.
    The mean is then subtracted by the constant C. If the pixel intensity is lower
    than the mean - C is set to 0, otherwise it is set to 255.
    """

    Constant = 2
    BlockSize = 9

    adaptive_gaussian = cv.adaptiveThreshold(
        frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, BlockSize, Constant
    )

    return adaptive_gaussian
    pass


def eliminate_noise(frame: Image, video: bool):
    """
    This applies a Gaussian Blur to the Image and only keeps the
    Very Bright Areas ig
    """
    if video:
        iter = 5
        Dilatekernel = np.ones((5, 5), np.uint8)
        blurKernel = (9, 9)
    else:
        iter = 2
        Dilatekernel = np.ones((3, 3), np.uint8)
        blurKernel = (5, 5)

    thresh = frame
    kernel = np.ones((5, 5), np.uint8)
    for i in range(iter):
        blur = cv.blur(thresh, blurKernel)
        ret, thresh = cv.threshold(blur, 205, 255, cv.THRESH_BINARY)
    thresh = cv.dilate(thresh, Dilatekernel, iterations=5)
    return thresh


def calculate_countours(frame: Image, video):
    if video:
        Areathres = 5000
    else:
        Areathres = 500
    contours, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    areas = [cv.contourArea(cnt) for cnt in contours]
    pairs = list(zip(areas, contours))
    pairs = list(filter(lambda x: x[0] > Areathres, pairs))
    pairs.sort(key=lambda x: x[0], reverse=True)
    ret = [x[1] for x in pairs[:5]]
    return ret


def calculate_centroids(contours):
    centroids = []
    for contour in contours:
        M = cv.moments(contour)
        try:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            continue
        coord = (cx, cy)
        centroids.append(coord)

    return centroids


def draw_centroids(centroids, frame):
    WHITE = (255, 255, 255)
    font = cv.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_thickness = 1
    radius = 5
    for center in centroids:
        cx, cy = center
        cv.putText(
            frame,
            f"center {cx, cy}",
            (cx + 40, cy),
            font,
            font_size,
            WHITE,
            font_thickness,
            cv.LINE_AA,
        )
        cv.circle(frame, center, radius, WHITE, -1)
