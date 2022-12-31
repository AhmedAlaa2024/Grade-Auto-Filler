import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from helper import *

# STUDENT_ID_LENGTH = 5
# NUM_QUESTIONS = 20
# NUM_CHOICES = 5
# MULTIPLE_CHOICE_IS_ALLOWED = True

ANSWERS_1 = [
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

ANSWERS_2 = [
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

def isFilled(image, A, thresh_1, thresh_2):
    Ax0 = A[0] - 5
    Ay0 = A[1] - 5
    Ax1 = A[0] + 5
    Ay1 = A[1] + 5
    circleImage = image[Ay0:Ay1,Ax0:Ax1]
    circleImage[circleImage < thresh_1] = 0
    sumPixels = np.sum(circleImage)

    
    if sumPixels < thresh_2:
        # cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        return True
    else:
        return False

def extractStudentID(id_length, bubblesList):
    STUDENT_ID = [0] * id_length
    bubblesList = np.array(bubblesList, dtype=object)
    bubblesList = np.transpose(bubblesList)
    # 10 for the number of decimal digits
    for i in range(id_length):
        if 'A' in bubblesList[i]:
            STUDENT_ID[0] = (i + 1) % 10
        if 'B' in bubblesList[i]:
            STUDENT_ID[1] = (i + 1) % 10
        if 'C' in bubblesList[i]:
            STUDENT_ID[2] = (i + 1) % 10
        if 'D' in bubblesList[i]:
            STUDENT_ID[3] = (i + 1) % 10
        if 'E' in bubblesList[i]:
            STUDENT_ID[4] = (i + 1) % 10

    s = [str(i) for i in STUDENT_ID]

    return "".join(s)


def bubble_sheet_autocorrect_1(image, ANSWERS, student_ID, NUM_QUESTIONS=20, NUM_ROWS=10, NUM_COLUMNS=10, NUM_CHOICES=5):
    image = skewCorrection(image)
    image = image[320:(image.shape[0]-200),100:(image.shape[1]-100)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 20, 70)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(edged, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,
               param2 = 15, minRadius = 13, maxRadius = 20)

    circles = []

    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        circles = []
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            circles.append((a, b, r))

    # Sort regarding to Y
    circles =  sorted(
        circles,
        key=lambda t: t[1]
    )

    # Sort every 15 circles regarding to X
    for i in range(NUM_ROWS):
        circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS] = sorted(circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS], key=lambda t: t[0])

    for circle in circles:
        a = circle[0]
        b = circle[1]
        r = circle[2]

    answersLeft = []
    answersRight = []

    print("\n========================= Answers Reports ========================= ")

    for i in range(0, len(circles) // NUM_CHOICES):
        A = circles[i*5]
        B = circles[(i*5)+1]
        C = circles[(i*5)+2]
        D = circles[(i*5)+3]
        E = circles[(i*5)+4]

        answers_vector = []
        isfound = False

        if isFilled(gray, A, 180, 10000):
            answers_vector.append('A')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (A[0], A[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, B, 180, 10000):
            answers_vector.append('B')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (B[0], B[1]), B[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (B[0], B[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, C, 180, 10000):
            answers_vector.append('C')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (C[0], C[1]), C[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (C[0], C[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, D, 180, 10000):
            answers_vector.append('D')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (D[0], D[1]), D[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (D[0], D[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, E, 180, 10000):
            answers_vector.append('E')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (E[0], E[1]), E[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (E[0], E[1]), 1, (255, 0, 0), 3)

        if not isfound:
            answers_vector.append(None)

        if (i % (NUM_COLUMNS // NUM_CHOICES) == 0):
            answersLeft.append(answers_vector)
        else:
            answersRight.append(answers_vector)


    bubbles = answersLeft + answersRight

    i = 1
    answers = []
    for bubble in bubbles:
        print("Q[{}]: {}".format(i, bubble))
        answers.append(bubble[0])
        i += 1

    print("\n========================= Wrong Answer Report ========================= ")
    wrong_answers = []
    correct = 0
    results = []
    for i in range(len(answers)):
        if (answers[i] == ANSWERS[i]):
            correct += 1
            results.append(True)
        else:
            wrong_answers.append((i+1, answers[i], ANSWERS[i]))
            print("Q[{}] is {} but it should be {}".format(i+1, answers[i], ANSWERS[i]))
            results.append(False)

    if correct == len(answers):
        print("None")
        
    print("\n========================= Conclusion Report ========================= ")
    print("Student ID: ", student_ID)
    print("Result: {}/{}".format(correct, NUM_QUESTIONS))

    show_images_1([image])

    return {
        "id": student_ID,
        "answers": results
    }

def bubble_sheet_autocorrect_2(image, ANSWERS, STUDENT_ID_LENGTH=5, NUM_QUESTIONS=50, NUM_ROWS=20, NUM_COLUMNS=15, NUM_CHOICES=5):
    image = skewCorrection(image)
    image = image[280:(image.shape[0]-150),100:(image.shape[1]-100)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 20, 70)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(edged, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 70,
                param2 = 15, minRadius = 10, maxRadius = 20)

    circles = []

    # Draw circles that are detected.
    if detected_circles is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        circles = []
    
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            circles.append((a, b, r))

    # Sort regarding to Y
    circles =  sorted(
        circles,
        key=lambda t: t[1]
    )

    # Sort every 15 circles regarding to X
    for i in range(NUM_ROWS):
        circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS] = sorted(circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS], key=lambda t: t[0])

    for circle in circles:
        a = circle[0]
        b = circle[1]
        r = circle[2]

    answersLeft = []
    answersMiddle = []
    answersRight = []

    print("\n========================= Answers Reports ========================= ")

    for i in range(0, len(circles) // NUM_CHOICES):
        A = circles[i*5]
        B = circles[(i*5)+1]
        C = circles[(i*5)+2]
        D = circles[(i*5)+3]
        E = circles[(i*5)+4]

        answers_vector = []
        isfound = False

        if isFilled(gray, A, 160, 3000):
            answers_vector.append('A')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (A[0], A[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, B, 160, 3000):
            answers_vector.append('B')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (B[0], B[1]), B[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (B[0], B[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, C, 160, 3000):
            answers_vector.append('C')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (C[0], C[1]), C[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (C[0], C[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, D, 160, 3000):
            answers_vector.append('D')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (D[0], D[1]), D[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (D[0], D[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, E, 160, 3000):
            answers_vector.append('E')
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (E[0], E[1]), E[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (E[0], E[1]), 1, (255, 0, 0), 3)

        if not isfound:
            answers_vector.append(None)

        if (i % (NUM_COLUMNS // NUM_CHOICES) == 0):
            answersLeft.append(answers_vector)
        elif (i % (NUM_COLUMNS // NUM_CHOICES) == 1):
            answersMiddle.append(answers_vector)
        else:
            answersRight.append(answers_vector)


    bubbles = answersLeft + answersMiddle + answersRight

    student_ID = extractStudentID(STUDENT_ID_LENGTH, bubbles[0:10])

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
    results = []
    for i in range(len(answers)):
        if (answers[i] == ANSWERS[i]):
            correct += 1
            results.append(True)
        else:
            wrong_answers.append((i+1, answers[i], ANSWERS[i]))
            print("Q[{}] is {} but it should be {}".format(i+1, answers[i], ANSWERS[i]))
            results.append(False)

    if correct == len(answers):
        print("None")
        
    print("\n========================= Conclusion Report ========================= ")
    print("Student ID: ", student_ID)
    print("Result: {}/{}".format(correct, NUM_QUESTIONS))

    show_images_1([image])

    return {
        "id": student_ID,
        "answers": results
    }

# image = cv2.imread("StudentAnswers/11.jpg")
image = cv2.imread("StudentAnswers/15.jpg")
# results = bubble_sheet_autocorrect_1(image, ANSWERS_1, "02141", NUM_QUESTIONS=20, NUM_ROWS=10, NUM_COLUMNS=10, NUM_CHOICES=5)
results = bubble_sheet_autocorrect_2(image, ANSWERS_2, 5, NUM_QUESTIONS=50, NUM_ROWS=20, NUM_COLUMNS=15, NUM_CHOICES=5)
print(results)