from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin
import skimage.io as io

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

# print("STEP 1: Edge Detection")
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break

# print("STEP 2: Find contours of paper")
# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
# print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
# cv2.waitKey(0)

img = warped
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
# cv2.imshow("Edged",cv2.resize(edges, (1200, 1600)))

binary = np.ones(img.shape)

lines = cv2.HoughLines(edges,1,np.pi/180,500)
print(lines.shape)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.line(binary,(x1,y1),(x2,y2),(0,0,0),5)

binary = np.float32(binary)
binary = cv2.cvtColor(binary,cv2.COLOR_BGR2GRAY)
print(binary)
binary[binary > 0.5] = 1
binary[binary <= 0.5] = 0
# binary = np.where(binary > 130, 255, 0)
# binary = binary_erosion(binary)
# binary = binary_dilation(binary)
# binary = thin(binary,1)
# cv2.imwrite('houghlines3.jpg',img)
# cv2.imshow("Hough",imutils.resize(img, height = 900))
# cv2.imshow("Hough-Binary",imutils.resize(binary, height = 900))
io.imshow(img)
# io.imshow(binary)
io.show()

# Sort by top to bottom and each row by left to right
# invert = 255 - thresh
binary = np.uint8(binary)
cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
# print(len(cnts))
grid_rows = []
row = []
for cell_points in cnts:
    if(len(cell_points) == 4):
        p1, p2, p3, p4 = cell_points
        p1 = p1[0]
        p2 = p2[0]
        p3 = p3[0]
        p4 = p4[0]
        cell = img[p1[1]:p2[1],p1[0]:p4[0]]
        cv2.imshow('result', cell)
        cv2.waitKey(0)

# for (i, c) in enumerate(cnts, 1):
#     area = cv2.contourArea(c)
#     if area < 50000:
#         row.append(c)
#         if i % 9 == 0:  
#             (cnts, _) = contours.sort_contours(row, method="left-to-right")
#             grid_rows.append(cnts)
#             row = []

# Iterate through each box
# for row in grid_rows:
#     for c in row:
#         mask = np.zeros(image.shape, dtype=np.uint8)
#         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
#         result = cv2.bitwise_and(image, mask)
#         result[mask==0] = 255
#         cv2.imshow('result', result)
#         cv2.waitKey(3000)
cv2.waitKey(0)