from commonfunctions import show_images
import math
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
	image: Original image
	return => The english name in that image
	"""
	text = pytesseract.image_to_string(image, lang='eng')
	return text.split('\n')[0]

# =============================================================================================


def getArabicName(image):
	"""
	image: Original image
	return => The arabic name in that image
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	threshImg = cv2.threshold(
			gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	text = pytesseract.image_to_string(threshImg, lang='ara')
	return text.split('\n')[0]

# =============================================================================================


def getCode(image):
	"""
	image: Original image
	return => The id number in that image
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
	return => True if the cell was right mark, false otherwise
	'''

	linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 50, None, 35, 10)
	numOfLines = 0

	if linesP is not None:
		for i in range(0, len(linesP)):
			(x1, y1, x2, y2) = linesP[i][0]
			if x1 != x2:
				angle = abs(math.atan((y2 - y1) / (x2 - x1)) * (180 / np.pi))

				if angle >= 20 and angle <= 70:
						numOfLines += 1

	return (numOfLines > 0)

# =============================================================================================


def detectVerticalLines(img):
	'''
	img: Preprocessed image given to detect if it was vertical line => number of lines
	return => Number of vertical lines if any exists
	'''

	verticalKernel = cv2.getStructuringElement(
			cv2.MORPH_RECT, (1, img.shape[0]//4))

	vertical = cv2.morphologyEx(
			img, cv2.MORPH_OPEN, verticalKernel, iterations=1)

	vertical = cv2.dilate(vertical, 
			cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

	verticalContours, _ = cv2.findContours(
			vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return len(verticalContours)

# =============================================================================================


def detectHorizontalLines(img):
	'''
	img: Preprocessed image given to detect if it was horizontal line => number of lines
	return => Number of horizontal lines if any exists
	'''

	horizontalKernel = cv2.getStructuringElement(
			cv2.MORPH_RECT, (img.shape[1]//4, 1))

	horizontal = cv2.morphologyEx(
			img, cv2.MORPH_OPEN, horizontalKernel, iterations=1)

	horizontal = cv2.dilate(horizontal, 
			cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

	horizontalContours, _ = cv2.findContours(
			horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	return len(horizontalContours)

# =============================================================================================


def detectBoxs(img, verticalLines, horizontalLines):
	'''
	img: Preprocessed image given to detect if it was box => True
	verticalLines: Number of vertical lines in this cell
	horizontalLines: Number of horizontal lines in this cell
	return => True if the cell was a box, false otherwise
	'''
	box = 0
	contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	for cnt in contours:
		size = cv2.contourArea(cnt)
		if size >= 1000:
			box += 1

	return box > 0 and verticalLines <= 2 and horizontalLines <= 2

# =============================================================================================


def detectQuestionMark(img, horizontalLines):
	'''
	img: Preprocessed image given to detect if it was question mark => True
	horizontalLines: Number of horizontal lines in this cell
	return => True if the cell was question mark, false otherwise
	'''
	detectedCircles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 80, param1=20,
			param2=9, minRadius=7, maxRadius=19)

	return (detectedCircles is not None
					and len(detectedCircles) == 1
					and horizontalLines == 0)

# =============================================================================================
# Detect Cells
# =============================================================================================


def enhanceCell(img):
	'''
	img: BGR cell that we want to enhance
	return => Enhanced image that is ready for detection phase
	'''

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	originalSize = img.shape

	img = cv2.adaptiveThreshold(
			img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

	img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)

	img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

	# ignore number of pixels up, down, right, and left
	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +
											cv2.THRESH_OTSU)[1][15:-14, 15:-14]

	img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)

	img = cv2.resize(img, (originalSize[1], originalSize[0]))

	img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY +
											cv2.THRESH_OTSU)[1]

	img = cv2.bitwise_not(img)

	# return cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)
	return cv2.morphologyEx(
			img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)


def detectCell(img):
	'''
	img: Original cell from the table
	return => Data of that cell after processing it
	'''
	# Preprocess the given image
	img_bin = enhanceCell(img)

	# Get the number of horizontal and vertical lines
	verticalLines = detectVerticalLines(img_bin)
	horizontalLines = detectHorizontalLines(img_bin)

	# Try to detect boxes
	box = detectBoxs(img_bin, verticalLines, horizontalLines)
	if box:
		return 0

	# Try to detect question mark
	questionMark = detectQuestionMark(img_bin, horizontalLines)
	if questionMark:
		return -1

	# Try to detect right mark
	rightMark = detectRightMark(img_bin)
	if rightMark:
		return 5

	# Try to detect minus
	if horizontalLines == 1:
		return 0
	elif horizontalLines != 0:
		return (5 - horizontalLines)

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
