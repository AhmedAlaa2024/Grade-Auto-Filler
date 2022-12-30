# import the necessary packages
from imutils.perspective import four_point_transform
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from helper import *

# load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread("AnswerSheets/7.jpg")
image = image[480:,:]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 60, 200)

thresholdedimage = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


circle_radius = 5
kernel1 = np.ones((2*circle_radius,2*circle_radius),dtype=int)
thresholdedimage = cv2.dilate(thresholdedimage, kernel1, iterations=1)
thresholdedimage = 255 - thresholdedimage

circle_radius = 25
kernel2 = np.ones((2*circle_radius,2*circle_radius),dtype=int)
answersImg = cv2.filter2D(src=thresholdedimage, ddepth=-1, kernel=kernel2)



titles = ["Original Image", "Convolved Image"]
images = [thresholdedimage, answersImg]

show_images(titles, images)