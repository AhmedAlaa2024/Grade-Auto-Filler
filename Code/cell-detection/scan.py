from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
edged[:,-1] = np.ones(edged.shape[0])
edged[:,0] = np.ones(edged.shape[0])
edged[-1,:] = np.ones(edged.shape[1])
edged[0,:] = np.ones(edged.shape[1])
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:6]

h,w = edged.shape
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	print(c,cv2.contourArea(approx), image.shape[0]*image.shape[1])
	print(approx)
	# cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	if len(approx) == 4:
		p1,p2,p3,p4 = approx
		p1,p2,p3,p4 = p1[0],p2[0],p3[0],p4[0]
		if (p1[0]<10 and p1[1]<10) or (w-p3[0]<10 and h-p3[1]<10) or (w-p2[0]<10 and p2[1]<10) or (p4[0]<10 and h-p4[1]<10):
			continue
		screenCnt = approx
		break

print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
cv2.imwrite('trans.jpg', warped)
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)