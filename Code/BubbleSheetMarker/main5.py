# import the necessary packages
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image, ImageDraw
from helper import *

ANSWERS = [
    'E',
    'A',
    'B',
    'D',
    'C',
    'C',
    'E',
    'A',
    'B',
    'D',
    'A',
    'B',
    'C',
    'D',
    'E',
    'E',
    'E',
    'B',
    'A',
    'D',
    'A',
    'A',
    'C',
    'B',
    'D',
    'E',
    'E',
    'B',
    'B',
    'B',
    'B',
    'C',
    'A',
    'B',
    'B',
    'E',
    'D',
    'D',
    'C',
    'A',
    'C',
    'B',
    'A',
    'B',
    'C',
    'D',
    'E',
    'D',
    'C',
    'B'
]

STUDENT_ID_LENGTH = 5
NUM_QUESTIONS = 50
NUM_CHOICES = 5
MULTIPLE_CHOICE_IS_ALLOWED = True

# load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread("StudentAnswers/15.jpg")
image = skewCorrection(image)
image = image[280:(image.shape[0]-150),100:(image.shape[1]-100)]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 20, 70)

# Apply Hough transform on the blurred image.
detected_circles_1 = cv2.HoughCircles(edged, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,
               param2 = 15, minRadius = 10, maxRadius = 20)

circles = []

# Draw circles that are detected.
if detected_circles_1 is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles_1 = np.uint16(np.around(detected_circles_1))
    circles = []
  
    for pt in detected_circles_1[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        circles.append((a, b, r))

# Sort regarding to Y
circles =  sorted(
    circles,
    key=lambda t: t[1]
)

# Sort every 15 circles regarding to X
for i in range(20):
    circles[(i*15):(i+1)*15] = sorted(circles[(i*15):(i+1)*15], key=lambda t: t[0])

for circle in circles:
    a = circle[0]
    b = circle[1]
    r = circle[2]

    # Draw the circumference of the circle.
    cv2.circle(image, (a, b), r, (0, 255, 0), 2)
    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(image, (a, b), 1, (0, 0, 255), 3)

answersLeft = []
answersMiddle = []
answersRight = []

print("\n========================= Answers Reports ========================= ")

for i in range(0, len(circles) // 5):
    A = circles[i*5]
    B = circles[(i*5)+1]
    C = circles[(i*5)+2]
    D = circles[(i*5)+3]
    E = circles[(i*5)+4]

    answers_vector = []
    isfound = False

    if isFilled(gray, A):
        answers_vector.append('A')
        isfound = True

        # Draw the circumference of the circle.
        cv2.circle(image, (A[0], A[1]), A[2], (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (A[0], A[1]), 1, (0, 255, 0), 3)

    if isFilled(gray, B):
        answers_vector.append('B')
        isfound = True

        # Draw the circumference of the circle.
        cv2.circle(image, (B[0], B[1]), B[2], (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (B[0], B[1]), 1, (0, 255, 0), 3)

    if isFilled(gray, C):
        answers_vector.append('C')
        isfound = True

        # Draw the circumference of the circle.
        cv2.circle(image, (C[0], C[1]), C[2], (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (C[0], C[1]), 1, (0, 255, 0), 3)

    if isFilled(gray, D):
        answers_vector.append('D')
        isfound = True

        # Draw the circumference of the circle.
        cv2.circle(image, (D[0], D[1]), D[2], (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (D[0], D[1]), 1, (0, 255, 0), 3)

    if isFilled(gray, E):
        answers_vector.append('E')
        isfound = True

        # Draw the circumference of the circle.
        cv2.circle(image, (E[0], E[1]), E[2], (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (E[0], E[1]), 1, (0, 255, 0), 3)

    if not isfound:
        answers_vector.append(None)

    if (i % 3 == 0):
        answersLeft.append(answers_vector)
    elif (i % 3 == 1):
        answersMiddle.append(answers_vector)
    else:
        answersRight.append(answers_vector)

bubbles = answersLeft + answersMiddle + answersRight

print("Student ID: ", extractStudentID(STUDENT_ID_LENGTH, bubbles[0:10]))
bubbles = bubbles[10:]
i = 1
answers = []
for bubble in bubbles:
    print("Q[{}]: {}".format(i, bubble))
    answers.append(bubble[0])
    i += 1

print("\n========================= Wrong Answer Report ========================= ")
wrong_answers = []
correct = 0
for i in range(len(answers)):
    if (answers[i] == ANSWERS[i]):
        correct += 1
    else:
        wrong_answers.append((i+1, answers[i], ANSWERS[i]))

if correct == len(answers):
    print("None")
    
print("\n========================= Conclusion Report ========================= ")
print("Result: {}/{}".format(correct, NUM_QUESTIONS))

for answer in wrong_answers:
    print("Q[{}] is {} but it should be {}".format(answer[0], answer[1], answer[2]))

titles = ["Answers Paper"]
images = [image]

show_images_1(images)