
# Road Lane Line Detector

# Educational project to learn about machine learning techniques and applications
# with computer vision

#  Import packages
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Canny edge detector
def canny_edge_detector(img):
    # Convert image to grayscale version
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use Gaussian filter to reduce noise
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


# Function for frame masking and finding region of interest (roi)
def roi(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])

    # Blank mask
    mask = np.zeros_like(img)

    # Fill color within roi defined by vertices
    cv2.fillPoly(mask, polygons, 255)

    # Return masked image
    return cv2.bitwise_and(img, mask)

# Helper function for find coordinates of road lane lines
def find_coordinates(img, line_params):
    slope, intercept = line_params
    y1 = img.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# Function for averaging and differentiating left and right line points
def avg_slope_intercept(img, lines):
    left_fit, right_fit = [], []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = find_coordinates(img, left_fit_avg)
    right_line = find_coordinates(img, right_fit_avg)
    return np.array([left_line, right_line])

# Fits coordinates to image and returns image with detected lines
def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x_l, y_l, x_r, y_r in lines:
            cv2.line(line_img, (x_l, y_l), (x_r, y_r), (255,0,0), 10)
    return line_img


# Grab video dataset and process
vid_capt = cv2.VideoCapture('./test2.mp4')

while(vid_capt.isOpened()):
    _, frame = vid_capt.read()
    canny_img = canny_edge_detector(frame)
    cropped_img = roi(canny_img)

    lines = cv2.HoughLinesP(cropped_img, 
                            rho=2,              # pixel resolution of r
                            theta=np.pi/180,    # degree resoultion of theta
                            threshold=100,      # min # of intersections to detect line
                            lines=np.array([]), # unused
                            minLineLength=40,   # min # of pts to form line
                            maxLineGap=5)       # max gap b/w 2 points in the line
    
    averaged_lines = avg_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    aggregate_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Results", aggregate_image)

    # 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    vid_capt.release()
    cv2.destroyAllWindows()
