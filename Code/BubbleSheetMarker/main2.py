# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from helper import show_images

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {
	0: 2,
	1: 3,
	2: 4,
	3: 3,
	4: 2,
	5: 1,
	6: 0,
	7: 1,
	8: 1,
	9: 0,
	10: 3,
	11: 4,
	12: 2,
	13: 2,
	14: 1,
	15: 0,
	16: 3,
	17: 3,
	18: 4,
	19: 2,
}

def get_rect_cnts(contours):
	rect_cnts = []
	for cnt in contours:
		# approximate the contour
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		print((peri, len(approx)))
		# if the approximated contour is a rectangle ...
		if len(approx) == 4:
			# append it to our list
			rect_cnts.append(approx)
	# sort the contours from biggest to smallest
	rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)
	
	return rect_cnts

# load the image, convert it to grayscale, blur it slightly, then find edges
image = cv2.imread("AnswerSheets/7.jpg")


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 60, 200)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresholdedimage = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
extractedContours = cv2.findContours(thresholdedimage.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
extractedContours = imutils.grab_contours(extractedContours)
questionContours = []

# loop over the contours
for c in extractedContours:
	# compute the bounding box of the contour, then use the
	# bounding box to derive the aspect ratio
	(x, y, w, h) = cv2.boundingRect(c)
	ratio = w / float(h)
	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
	if w >= 20 and h >= 20 and ratio >= 0.915 and ratio <= 1.1:
		questionContours.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionContours = contours.sort_contours(questionContours,
	method="top-to-bottom")[0]
correct = 0

rect_cnts = get_rect_cnts(questionContours)

color = (0, 255, 0)
# draw the outline of the correct answer on the test
cv2.drawContours(image, questionContours, -1, color, 3)

show_images(
	["Original image"],
	[image]
)






def temp():
		# Each question has 5 possible answers, to loop over the
	# question in batches of 5
	for (q, i) in enumerate(np.arange(0, len(questionContours), 5)):
		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		subContours = contours.sort_contours(questionContours[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(subContours):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresholdedimage.shape, dtype="uint8")
			snapshot.append(mask.copy())
			cv2.drawContours(mask, [c], -1, 255, -1)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresholdedimage, thresholdedimage, mask=mask)
			total = cv2.countNonZero(mask)
			num_active_pixels = 0
			for i in range(mask.shape[0]):
				for j in range(mask.shape[1]):
					if mask[i][j] == 0:
						num_active_pixels += 1
			
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if bubbled is None or total > bubbled[0]:
				snapshot.append(mask.copy())
				bubbled = (total, j)

		# initialize the contour color and the index of the
		# *correct* answer
		color = (0, 0, 255)
		k = ANSWER_KEY[q]
		# check to see if the bubbled answer is correct
		if k == bubbled[1]:
			color = (0, 255, 0)
			correct += 1

	# print(len(bubbled))
	color = (0, 255, 0)
	# draw the outline of the correct answer on the test
	cv2.drawContours(image, questionContours, -1, color, 3)