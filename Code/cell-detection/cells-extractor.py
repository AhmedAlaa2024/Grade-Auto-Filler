from transform import four_point_transform
import numpy as np
import argparse
import cv2
import imutils
from imutils import contours

def is_noisy(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.09
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr

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
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:6]

h,w = edged.shape
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		p1,p2,p3,p4 = approx
		p1,p2,p3,p4 = p1[0],p2[0],p3[0],p4[0]
		if (p1[0]<10 and p1[1]<10) or (w-p3[0]<10 and h-p3[1]<10) or (w-p2[0]<10 and p2[1]<10) or (p4[0]<10 and h-p4[1]<10):
			continue
		screenCnt = approx
		break

# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


img = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
# warped = (warped > T).astype("uint8") * 255
# cv2.imshow("Original", imutils.resize(orig, height = 650))
# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
# cv2.waitKey(0)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# applyMedian = is_noisy(img)
# print(applyMedian)
# if(applyMedian):
#     gray = cv2.medianBlur(gray, 3)
edges = cv2.Canny(gray,25,25,apertureSize = 3)
cv2.imshow("Edged",cv2.resize(edges, (1200, 1600)))
cv2.waitKey(0)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
binary = np.ones(img.shape)

cv2.namedWindow("Hough_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hough_Window", 900, 900)
cv2.namedWindow("Hough-Binary_Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hough-Binary_Window", 900, 900)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
detected_lines_H = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cntsH = cv2.findContours(detected_lines_H, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsH = cntsH[0] if len(cntsH) == 2 else cntsH[1]
binaryH = np.zeros(img.shape)
for c in cntsH:
    cv2.drawContours(binaryH, [c], -1, (255,255,255), 2)
# cv2.namedWindow("H-Morph-Window", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("H-Morph-Window", 900, 900)
# cv2.imshow("H-Morph-Window", binaryH)
# cv2.waitKey(0)
binaryH = binaryH[:,:,0]
binaryH = binaryH.astype('uint8')

lines = cv2.HoughLines(binaryH,1,np.pi/180,530)
linesH = []
for line in lines:
    for rho,theta in line:
        if theta != 0:
            linesH.append((rho, theta))
linesH.sort()
for rho,theta in linesH:
    if(theta > 1.58 or theta < 1.57):
        continue
    # print(rho, theta)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 3000*(-b))
    y1 = int(y0 + 3000*(a))
    x2 = int(x0 - 3000*(-b))
    y2 = int(y0 - 3000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.line(binary,(x1,y1),(x2,y2),(0,0,0),5)
    # cv2.imshow("Hough_Window",img)
    # cv2.imshow("Hough-Binary_Window",binary)
    # cv2.waitKey(0)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
detected_lines_V = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cntsV = cv2.findContours(detected_lines_V, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntsV = cntsV[0] if len(cntsV) == 2 else cntsV[1]
binaryV = np.zeros(img.shape)
for c in cntsV:
    cv2.drawContours(binaryV, [c], -1, (255,255,255), 2)
# cv2.namedWindow("V-Morph-Window", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("V-Morph-Window", 900, 900)
# cv2.imshow("V-Morph-Window", binaryV)
# cv2.waitKey(0)
binaryV = binaryV[:,:,0]
binaryV = binaryV.astype('uint8')

lines = cv2.HoughLines(binaryV,1,np.pi/180,360)
linesV = []
for line in lines:
    for rho,theta in line:
        if theta == 0:
            linesV.append((rho, theta))
linesV.sort()
for rho,theta in linesV:
    # print(rho)
    a = np.cos(theta)      
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 3000*(-b))
    y1 = int(y0 + 3000*(a))
    x2 = int(x0 - 3000*(-b))
    y2 = int(y0 - 3000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
    cv2.line(binary,(x1,y1),(x2,y2),(0,0,0),5)
    # cv2.imshow("Hough_Window",img)
    # cv2.imshow("Hough-Binary_Window",binary)
    # cv2.waitKey(0)

binary = np.float32(binary)
binary = cv2.cvtColor(binary,cv2.COLOR_BGR2GRAY)
binary[binary > 0.5] = 1
binary[binary <= 0.5] = 0
cv2.imshow("Hough_Window",img)
cv2.imshow("Hough-Binary_Window",binary)
cv2.waitKey(0)

binary = np.uint8(binary)
cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
cells = []
for cell_points in cnts:
    if(len(cell_points) == 4):
        p1, p2, p3, p4 = cell_points
        p1 = p1[0]
        p2 = p2[0]
        p3 = p3[0]
        p4 = p4[0]
        # print(cell_points)
        h,w,c = img.shape
        # very top cell || very bottom cell || very left cell || very right cell || min cell height || min cell width
        if p1[0] < 20 or h-p3[1] < 20 or p1[1] < 20 or w-p4[0] < 10 or p2[1]-p1[1] < h*50/3532 or p3[0]-p2[0] < w*50/2638:
            continue
        cell = img[p1[1]:p2[1],p1[0]:p4[0]]
        cells.append(cell)
        cv2.imshow('result', cell)
        cv2.waitKey(0)