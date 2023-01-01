import numpy as np
import cv2
import os
from cellsExtractionPhase import skewCorrection


def show_images(titles, images, wait=True):
    """Display multiple images with one line of code"""

    for (title, image) in zip(titles, images):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL) 
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def isFilled(image, A, thresh_1, thresh_2):
    Ax0 = A[0] - 5
    Ay0 = A[1] - 5
    Ax1 = A[0] + 5
    Ay1 = A[1] + 5
    circleImage = image[Ay0:Ay1,Ax0:Ax1]
    circleImage[circleImage < thresh_1] = 0
    sumPixels = np.sum(circleImage)

    
    if sumPixels < thresh_2:
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

# bubble_sheet_autocorrect_1(saveImagesDir, id, numQuestions, numRow, numCol, numChoices)
def bubble_sheet_autocorrect_1(imagePath, modelAnswer, wantToSaveImage, saveImagesDir, student_ID, NUM_QUESTIONS=20, numRow=10, NUM_COLUMNS=10, NUM_CHOICES=5):
    image = cv2.imread(imagePath + f"/{student_ID}.jpg")
    image = skewCorrection(image)
    original = image.copy()
    image = image[320:(image.shape[0]-200),100:(image.shape[1]-100)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    for i in range(numRow):
        circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS] = sorted(circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS], key=lambda t: t[0])

    for circle in circles:
        a = circle[0]
        b = circle[1]
        r = circle[2]

    answersLeft = []
    answersRight = []

    markedLeft = []
    markedRight = []

    markedCircles = []
    circle = 0
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
            circle = A

        if isFilled(gray, B, 180, 10000):
            answers_vector.append('B')
            isfound = True
            circle = B

        if isFilled(gray, C, 180, 10000):
            answers_vector.append('C')
            isfound = True
            circle = C

        if isFilled(gray, D, 180, 10000):
            answers_vector.append('D')
            isfound = True

            circle = D

        if isFilled(gray, E, 180, 10000):
            answers_vector.append('E')
            isfound = True

            circle = E

        if not isfound:
            answers_vector.append(None)

        if (i % (NUM_COLUMNS // NUM_CHOICES) == 0):
            answersLeft.append(answers_vector)
            markedLeft.append(circle)
        else:
            answersRight.append(answers_vector)
            markedRight.append(circle)


    bubbles = answersLeft + answersRight
    markedCircles = markedLeft + markedRight

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
        circle = markedCircles[i]
        if (answers[i] == modelAnswer[i]):
            correct += 1
            results.append(True)

            # Draw the circumference of the circle.
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (circle[0], circle[1]), 1, (255, 0, 0), 3)
        else:
            wrong_answers.append((i+1, answers[i], modelAnswer[i]))
            print("Q[{}] is {} but it should be {}".format(i+1, answers[i], modelAnswer[i]))
            results.append(False)

            # Draw the circumference of the circle.
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (circle[0], circle[1]), 1, (255, 0, 0), 3)

    if correct == len(answers):
        print("None")
        
    print("\n========================= Conclusion Report ========================= ")
    print("Student ID: ", student_ID)
    print("Result: {}/{}".format(correct, NUM_QUESTIONS))
    print("Marked Paper is saved in: ", str(saveImagesDir + '/' + student_ID + ".jpg"))

    original[320:(original.shape[0]-200),100:(original.shape[1]-100)] = image
    if wantToSaveImage:
        cv2.imwrite(str(saveImagesDir + '/' + student_ID + ".jpg"), original)

    return {
        "id": student_ID,
        "answers": results
    }

def bubble_sheet_autocorrect_2(imagePath, modelAnswer, wantToSaveImage, saveImagesDir, STUDENT_ID_LENGTH=5, NUM_QUESTIONS=50, numRow=20, NUM_COLUMNS=15, NUM_CHOICES=5):
    image = cv2.imread(imagePath)
    image = skewCorrection(image)
    original = image.copy()
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
    for i in range(numRow):
        circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS] = sorted(circles[(i*NUM_COLUMNS):(i+1)*NUM_COLUMNS], key=lambda t: t[0])

    for circle in circles:
        a = circle[0]
        b = circle[1]
        r = circle[2]

    answersLeft = []
    answersMiddle = []
    answersRight = []

    markedLeft = []
    markedMiddle = []
    markedRight = []

    markedCircles = []

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
            circle = A
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (A[0], A[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, B, 160, 3000):
            answers_vector.append('B')
            circle = B
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (B[0], B[1]), B[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (B[0], B[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, C, 160, 3000):
            answers_vector.append('C')
            circle = C
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (C[0], C[1]), C[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (C[0], C[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, D, 160, 3000):
            answers_vector.append('D')
            circle = D
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (D[0], D[1]), D[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (D[0], D[1]), 1, (255, 0, 0), 3)

        if isFilled(gray, E, 160, 3000):
            answers_vector.append('E')
            circle = E
            isfound = True

            # Draw the circumference of the circle.
            cv2.circle(image, (E[0], E[1]), E[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (E[0], E[1]), 1, (255, 0, 0), 3)

        if not isfound:
            answers_vector.append(None)

        if (i % (NUM_COLUMNS // NUM_CHOICES) == 0):
            markedLeft.append(circle)
            answersLeft.append(answers_vector)
        elif (i % (NUM_COLUMNS // NUM_CHOICES) == 1):
            markedMiddle.append(circle)
            answersMiddle.append(answers_vector)
        else:
            markedRight.append(circle)
            answersRight.append(answers_vector)

    bubbles = answersLeft + answersMiddle + answersRight
    markedCircles = markedLeft + markedMiddle + markedRight

    student_ID = extractStudentID(STUDENT_ID_LENGTH, bubbles[0:10])

    bubbles = bubbles[10:]
    markedCircles = markedCircles[10:]
    
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
        circle = markedCircles[i]
        if (answers[i] == modelAnswer[i]):
            correct += 1
            results.append(True)

            # Draw the circumference of the circle.
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (circle[0], circle[1]), 1, (255, 0, 0), 3)
        else:
            wrong_answers.append((i+1, answers[i], modelAnswer[i]))
            print("Q[{}] is {} but it should be {}".format(i+1, answers[i], modelAnswer[i]))
            results.append(False)

            # Draw the circumference of the circle.
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(image, (circle[0], circle[1]), 1, (255, 0, 0), 3)

    if correct == len(answers):
        print("None")
        
    print("\n========================= Conclusion Report ========================= ")
    print("Student ID: ", student_ID)
    print("Result: {}/{}".format(correct, NUM_QUESTIONS))
    print("Marked Paper is saved in: ", str(saveImagesDir + '/' + student_ID + ".jpg"))

    original[280:(original.shape[0]-150),100:(original.shape[1]-100)] = image

    if wantToSaveImage:
        cv2.imwrite(str(saveImagesDir + '/' + student_ID + ".jpg"), original)

    return {
        "id": student_ID,
        "answers": results
    }


def bubbleSheetAutoCorrector(config):
    studentAnswerPaperPath = config["studentAnswerPaperPath"]
    numStudents = config["numStudents"]
    numQuestions = config["numQuestions"]
    numChoices = config["numChoices"]
    idExist = config["idExist"]
    IdLen = config["IdLen"]
    modelAnsPath = config["modelAnsFile"]
    idListPath = config["idList"]
    wantToSaveImage = config["saveImages"]
    saveImagesDir = config["saveImagesDir"]
    numCol = config["numCol"]
    numRow = config["numRow"]

    results = []

    modelAnswer = []
    file = open(modelAnsPath, 'r')
    for line in file:
        modelAnswer.append(line.split("\n")[0])
    file.close()

    if not idExist:
        idList = []
        file = open(idListPath, 'r')
        for line in file:
            idList.append(line.split("\n")[0])
        file.close()

        if numStudents != len(idList):
            print("Number of students doesn't match the number of IDs in the list!")
            raise SystemExit(2)

        for id in idList:
            result = bubble_sheet_autocorrect_1(studentAnswerPaperPath, modelAnswer, wantToSaveImage, saveImagesDir, id, numQuestions, numRow, numCol, numChoices)
            results.append(result)
    else:

        # get all the image names
        all_images = os.listdir(studentAnswerPaperPath)

        if numStudents != len(all_images):
            print("Number of students doesn't match the number of papers in the SAMPLE_PATH directory!")
            raise SystemExit(2)

        # iterate over the image names, get the label
        for image in all_images:
            image_path = f"{studentAnswerPaperPath}/{image}"
            
            result = bubble_sheet_autocorrect_2(image_path, modelAnswer, wantToSaveImage, saveImagesDir, IdLen, numQuestions, numRow, numCol, numChoices)
            results.append(result)

    return results