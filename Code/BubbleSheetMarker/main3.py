# import the necessary packages
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from helper import *

ANSWERS = [
    'C',
    'D',
    'E',
    'D',
    'C',
    'B',
    'A',
    'B',
    'B',
    'A',
    'D',
    'E',
    'C',
    'C',
    'B',
    'A',
    'D',
    'D',
    'E',
    'C'
]

# load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread("AnswerSheets/11.jpg")
image = skewCorrection(image)
image = image[320:(image.shape[0]-220),100:(image.shape[1]-100)]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 20, 70)

# Apply Hough transform on the blurred image.
detected_circles_1 = cv2.HoughCircles(edged, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,
               param2 = 20, minRadius = 10, maxRadius = 20)

thresholdedimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Dilation with circular kernel with radius = 5
circle_radius = 1
kernel1 = np.ones((2*circle_radius,2*circle_radius),dtype=int)
thresholdedimage = cv2.dilate(thresholdedimage, kernel1, iterations=1)
thresholdedimage = 255 - thresholdedimage

# convolution with circular kernel with radius = 25
circle_radius = 5
kernel2 = np.ones((2*circle_radius,2*circle_radius),dtype=int)
answersImg = cv2.filter2D(src=thresholdedimage, ddepth=-1, kernel=kernel2)

# Erosion with circular kernel with radius = 5
circle_radius = 10
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*circle_radius,2*circle_radius))
answersImg = cv2.erode(answersImg, kernel3, iterations=1)

answerContours = cv2.findContours(255-answersImg.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
answerContours = imutils.grab_contours(answerContours)
print("Number of answers =", len(answerContours))
color = (0, 0, 255)
# draw the outline of the correct answer on the test
# cv2.drawContours(image, answerContours, -1, color, 3)

circles = []

# Draw circles that are detected.
if detected_circles_1 is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles_1 = np.uint16(np.around(detected_circles_1))
    circles = []
  
    for pt in detected_circles_1[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        circles.append((a, b, r))

print("Number of circles =", len(circles))

# Sort regarding to Y
circles =  sorted(
    circles,
    key=lambda t: t[1]
)

# Sort every 10 circles regarding to X
for i in range(10):
    circles[(i*10):(i+1)*10] = sorted(circles[(i*10):(i+1)*10], key=lambda t: t[0])

for circle in circles:
    a = circle[0]
    b = circle[1]
    r = circle[2]

    # Draw the circumference of the circle.
    cv2.circle(image, (a, b), r, (0, 255, 0), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

answersCircles = []
correct = 0

# loop over the contours
for c in answerContours:
    # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    answersCircles.append((x, y, w, h, c))

answersCircles = sorted(answersCircles, key=lambda t: t[1])

# Sort every 2 answer circles regarding to X
for i in range(10):
    answersCircles[(i*2):(i+1)*2] = sorted(answersCircles[(i*2):(i+1)*2], key=lambda t: t[0])

answersLower = []
answersHigher = []

print(" ========================= Answers Reports ========================= ")

for (i, circle) in enumerate(answersCircles):
    A = circles[i*5]
    B = circles[(i*5)+1]
    C = circles[(i*5)+2]
    D = circles[(i*5)+3]
    E = circles[(i*5)+4]

    x = int(circle[0] + (circle[2]/2))
    y = int(circle[1] + (circle[3]/2))

    # M = cv2.moments(circle[4])
    # x = int(M["m10"] / M["m00"])
    # y = int(M["m01"] / M["m00"])

    # # Draw the circumference of the circle.
    # cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(image, (A[0], A[1]), 1, (0, 0, 255), 3)

    # # Draw the circumference of the circle.
    # cv2.circle(image, (B[0], B[1]), B[2], (0, 255, 0), 2)
    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(image, (B[0], B[1]), 1, (0, 0, 255), 3)

    # # Draw the circumference of the circle.
    # cv2.circle(image, (C[0], C[1]), C[2], (0, 255, 0), 2)
    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(image, (C[0], C[1]), 1, (0, 0, 255), 3)

    # # Draw the circumference of the circle.
    # cv2.circle(image, (D[0], D[1]), D[2], (0, 255, 0), 2)
    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(image, (D[0], D[1]), 1, (0, 0, 255), 3)

    # # Draw the circumference of the circle.
    # cv2.circle(image, (E[0], E[1]), E[2], (0, 255, 0), 2)
    # # Draw a small circle (of radius 1) to show the center.
    # cv2.circle(image, (E[0], E[1]), 1, (0, 0, 255), 3)

    # Draw the circumference of the circle.
    cv2.circle(image, (x, y), 15, (0, 0, 255), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(image, (x, y), 1, (0, 255, 0), 3)

    if (x <= B[0]):
        print("Q", (i+1), "Answer: A")

        if (i % 2 == 0):
            answersLower.append('A')
        else:
            answersHigher.append('A')
    elif ((x > A[0]) and (x < C[0])):
        print("Q", (i+1), "Answer: B")
                
        if (i % 2 == 0):
            answersLower.append('B')
        else:
            answersHigher.append('B')
    elif ((x > B[0]) and (x < D[0])):
        print("Q", (i+1), "Answer: C")
                
        if (i % 2 == 0):
            answersLower.append('C')
        else:
            answersHigher.append('C')
    elif ((x > C[0]) and (x < E[0])):
        print("Q", (i+1), "Answer: D")
                
        if (i % 2 == 0):
            answersLower.append('D')
        else:
            answersHigher.append('D')
    elif (x > D[0]):
        print("Q", (i+1), "Answer: E")
                
        if (i % 2 == 0):
            answersLower.append('E')
        else:
            answersHigher.append('E')

print(" ========================= Wrong Answer Report ========================= ")
answers = answersLower + answersHigher
wrong_answers = []
correct = 0
for i in range(len(answers)):
    if (answers[i] == ANSWERS[i]):
        correct += 1
    else:
        wrong_answers.append((i+1, answers[i], ANSWERS[i]))

if correct == len(answers):
    print("None")
    
print(" ========================= Conclusion Report ========================= ")
print("Number of correct answers: {}/20".format(correct))

for answer in wrong_answers:
    print("Q[{}] is {} but it should be {}".format(answer[0], answer[1], answer[2]))

titles = ["Detected Circle 1", "Answers Image"]
images = [gray, image]

show_images_1(images)