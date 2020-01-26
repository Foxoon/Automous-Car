import cv2
import numpy as np


def imgproc(img):
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = 5
    #gaussian filter to remove noises
    smooth = cv2.GaussianBlur(grey, (kernel, kernel), 0)
    canny = cv2.Canny(smooth, 50, 150)
    return canny

#region of interest
def roi(canny):
    height = canny.shape[0]
    width = canny.shape[1]

    mask = np.zeros_like(canny)

    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height), ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    maskedImage = cv2.bitwise_and(canny, mask)
    return maskedImage


def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 3 / 5)  # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:  # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines

def displayLines(img, lines):
    lineImage = np.zeros_like(img)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lineImage, (x1,y1),(x2,y2),(255,0,0),10)
    return lineImage


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    postprocImage = imgproc(frame)
    croppedImage = roi(postprocImage)

    lines = cv2.HoughLinesP(croppedImage, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    #averagedLines = average_slope_intercept(frame, lines)
    linesImage = displayLines(frame, lines)
    mergedImage = cv2.addWeighted(frame, 0.7, linesImage, 1, 1)

    cv2.imshow("result", mergedImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()