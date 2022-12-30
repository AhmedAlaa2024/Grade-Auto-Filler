import cv2
import numpy as np
import skimage.io as io
from skimage.filters import sobel
from skimage.morphology import closing, opening, binary_erosion, binary_dilation, erosion
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.draw import polygon
import matplotlib.pyplot as plt

from helper import *

green = (0, 255, 0) # green color
red = (0, 0, 255) # red color
white = (255, 255, 255) # white color

def get_contours(img):
    contours = find_contours(img, 0.8)
    contours_points = np.zeros(img.shape)
    # cv2.drawContours(img, contours, -1, green, 3)

    for c in contours:
        Xmax,Xmin,Ymax,Ymin = max(c[:,1]), min(c[:,1]), max(c[:,0]), min(c[:,0])
        rr,cc = polygon([Ymin, Ymax, Ymax, Ymin], [Xmin, Xmin, Xmax, Xmax], shape=img.shape)
        contours_points[rr, cc] = 1

    return contours_points

def FilterContours(contourImg, area_ratio = None):
    contours = find_contours(contourImg, 0.8)
    contours_points = np.zeros(contourImg.shape)
    Areas = []
    # Save the (x,y) to not repeat the loops
    Xs = []
    Ys = []

    for c in contours:
        Xmax, Xmin, Ymax, Ymin = max(c[:,1]), min(c[:,1]), max(c[:,0]), min(c[:,0])
        Xs.append([Xmin, Xmax])
        Ys.append([Ymin, Ymax])
        area = (Xmax - Xmin) * (Ymax - Ymin)
        Areas.append(area)

    variance = np.sqrt(np.var(np.array(Areas)))
    mean = np.mean(np.array(Areas))

    if (area_ratio is not None):
        variance = area_ratio * mean

    for c in range(len(contours)):
        # Pass specific contours
        if abs(Areas[c] - mean) <= variance:
            rr,cc = polygon([Ys[c][0], Ys[c][1], Ys[c][1], Ys[c][0]],
                            [Xs[c][0], Xs[c][0], Xs[c][1], Xs[c][1]], shape=img.shape)
            contours_points[rr,cc] = 1
    return contours_points

def extract_choices(img):
    img_copy = img.copy()[420:(img.shape[0] - 420),:]
    
    img_copy = cv2.GaussianBlur(img_copy, (5,5), 0)
    edges = cv2.Canny(img_copy,75,200)

    return edges

img = cv2.imread('AnswerSheets/6.jpg')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = extract_choices(img).astype('uint8')
contours_points_1 = get_contours(img)
contours_points_2 = FilterContours(contours_points_1, 1)
contours_points_3 = FilterContours(contours_points_2, 1)
contours_points_4 = FilterContours(contours_points_3, 0.3)
final_img = get_contours(contours_points_4)
contours = find_contours(final_img, 0.8)
show_images(["Original Image", "Countoured Image"], [img, contours_points_4])

cv2.waitKey(0)
cv2.destroyAllWindows()