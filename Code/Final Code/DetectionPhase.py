import numpy as np
import cv2
import pytesseract
import os
os.environ["TESSDATA_PREFIX"] = r'C:\Users\iiBesh00\AppData\Local\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\iiBesh00\AppData\Local\Tesseract-OCR\tesseract.exe'

# =============================================================================================
# Detect names, codes and numeric values usign OCR
# =============================================================================================


def getEnglishName(image):
    """
    This function will handle the core OCR processing of getting english name.
    """
    text = pytesseract.image_to_string(image, lang='eng')
    return text.split('\n')[0]

# =============================================================================================


def getArabicName(image):
    """
    This function will handle the core OCR processing of getting arabic name.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshImg = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(threshImg, lang='ara')
    return text.split('\n')[0]

# =============================================================================================


def getCode(image):
    """
    This function will handle the core OCR processing of getting id number.
    """
    text = pytesseract.image_to_string(image, config='digits')
    return text.split('\n')[0]

# =============================================================================================


def detectNumericValues(img):
    '''
    img: Original image
    return => The number in that image
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (600, 400))
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    img_bin = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_bin = cv2.bitwise_not(img_bin)
    img_bin[0:50, :] = 0
    img_bin[-50:, :] = 0
    img_bin[:, 0:50] = 0
    img_bin[:, -50:] = 0

    num = pytesseract.image_to_string(img_bin, config=("-c tessedit"
                                                       "_char_whitelist=0123456789"
                                                       " --psm 10"
                                                       " -l osd"
                                                       " "))
    return num.split("\n")[0]

# =============================================================================================
# Detect Symbols using vanela image processing
# =============================================================================================


def detectRightMark(img):
    '''
    img: Preprocessed image given to detect if it was right mark => True
    '''

    erodeKernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, erodeKernel, iterations=1)

    kernel = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    diagonal = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations=1)
    diagonalContours, _ = cv2.findContours(
        diagonal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diagonalResult = len(diagonalContours)

    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical = cv2.morphologyEx(
        erosion, cv2.MORPH_OPEN, verticalKernel, iterations=1)
    verticalContours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    verticalResult = len(verticalContours)

    return (verticalResult == 1 and diagonalResult == 1)

# =============================================================================================


def detectVerticalLines(img):
    '''
    img: Preprocessed image given to detect if it was vertical line => number of lines
    '''

    verticalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, verticalKernel, iterations=1)
    verticalContours, _ = cv2.findContours(
        vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(verticalContours)

# =============================================================================================


def detectHorizontalLines(img):
    '''
    img: Preprocessed image given to detect if it was horizontal line => number of lines
    '''

    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    horizontal = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, horizontalKernel, iterations=1)

    horizontalContours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(horizontalContours)

# =============================================================================================


def detectBoxs(img):
    '''
    img: Preprocessed image given to detect if it was box => True
    '''
    verticals = detectVerticalLines(img)
    horizontals = detectHorizontalLines(img)

    return (verticals == 2 and horizontals == 2)

# =============================================================================================


def detectQuestionMark(img):
    '''
    img: Preprocessed image given to detect if it was question mark => True
    '''
    detectedCircles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 80, param1=20,
                                       param2=9, minRadius=10, maxRadius=17)

    return detectedCircles is not None and len(detectedCircles) == 1

# =============================================================================================
# Detect Cells
# =============================================================================================


def detectCell(img):
    '''
    img: Original cell from the table
    return => Data of that cell after processing it
    '''
    # Preprocess the given image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    img_bin = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_bin = cv2.bitwise_not(img_bin)
    img_bin[0:7, :] = 0
    img_bin[-7:, :] = 0
    img_bin[:, 0:20] = 0
    img_bin[:, -20:] = 0

    # Try to detect right mark
    rightMark = detectRightMark(img_bin)
    if rightMark:
        return 5

    # Try to detect boxes
    box = detectBoxs(img_bin)
    if box:
        return 0

    # Try to detect question mark
    questionMark = detectQuestionMark(img_bin)
    if questionMark:
        return -1

    # Try to detect minus
    horizontalLines = detectHorizontalLines(img_bin)
    if horizontalLines == 1:
        return 0
    elif horizontalLines != 0:
        return (5 - horizontalLines)

    verticalLines = detectVerticalLines(img_bin)
    if verticalLines != 0:
        return verticalLines

    # Else => Empty cell
    return -2

# =============================================================================================
# Detectin phase function
# =============================================================================================


def detectionPhase(images):
    '''
    images: Array of cells ready to be detected
    return => Data ready to be exported to excel sheet
    '''

    # images[i] => 3 | images[i+1] => 2 | images[i+2] => 1
    # images[i+3] => English Name
    # images[i+4] => Student Name
    # images[i+5] => Code

    finalData = []
    for i in range(0, len(images), 6):
        thirdCell = detectCell(images[i])
        secondCell = detectCell(images[i+1])
        firstCell = detectNumericValues(images[i+2])

        englishName = getEnglishName(images[i+3])
        studentName = getArabicName(images[i+4])
        code = getCode(images[i+5])

        data = {
            "Code": code,
            "Student Name": studentName,
            "English Name": englishName,
            "1": firstCell,
            "2": secondCell,
            "3": thirdCell
        }
        finalData.append(data)

    return finalData
