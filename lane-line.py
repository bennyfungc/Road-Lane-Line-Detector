
# Road Lane Line Detector

# Educational project to learn about machine learning techniques and applications
# with computer vision

#  Import packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import matplotlib.image as mpimg
import math
from moviepy.editor import VideoFileClip



# Function for frame masking and finding region of interest (roi) defined by
# vertices input
# Returns original image with everything masked except roi
def roi(img, vertices):

    # Blank mask
    mask = np.zeros_like(img)

    # Fill color within roi defined by vertices
    cv2.fillPoly(mask, vertices, 255)

    # Return masked image
    return cv2.bitwise_and(img, mask)


# Function for line/edge detection using Hough Transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # todo: function to draw lines on line_img

    return line_img