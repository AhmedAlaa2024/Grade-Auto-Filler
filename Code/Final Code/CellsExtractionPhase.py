import numpy as np
import cv2
import imutils
from imutils import contours

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]
    ], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def skewCorrection(image):
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
    img = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return img

def detectLines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
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

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detected_lines_V = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cntsV = cv2.findContours(detected_lines_V, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsV = cntsV[0] if len(cntsV) == 2 else cntsV[1]
    binaryV = np.zeros(img.shape)
    for c in cntsV:
        cv2.drawContours(binaryV, [c], -1, (255,255,255), 2)
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
    return binary

def extractCells(binary, img):
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
            h,w,c = img.shape
            # very top cell || very bottom cell || very left cell || very right cell || min cell height || min cell width
            if p1[0] < 20 or h-p3[1] < 20 or p1[1] < 20 or w-p4[0] < 10 or p2[1]-p1[1] < h*50/3532 or p3[0]-p2[0] < w*50/2638:
                continue
            cell = img[p1[1]:p2[1],p1[0]:p4[0]]
            cells.append(cell)
            cv2.imshow('result', cell)
            cv2.waitKey(0)
    return cells

def main(imgPath):
    image = cv2.imread(imgPath)

    skewCorrected = skewCorrection(image)

    binary = detectLines(skewCorrected)

    cells = extractCells(binary, skewCorrected)

    return cells

cells = main('../Samples/Samples/15.jpg')