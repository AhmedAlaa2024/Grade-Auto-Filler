import cv2
import numpy as np

img = cv2.imread('13.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow("Edged",cv2.resize(edges, (1200, 1600)))

bi = cv2.bilateralFilter(gray, 5, 75, 75)
dst = cv2.cornerHarris(bi, 2, 3, 0.04)
mask = np.zeros_like(gray)
mask[dst>0.01*dst.max()] = 255
cv2.imshow('mask', cv2.resize(mask, (1200, 1600)))

lines = cv2.HoughLines(edges,1,np.pi/180,400)
for line in lines:
    for rho,theta in line:
        # print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

# cv2.imwrite('houghlines3.jpg',img)
cv2.imshow("Hough",cv2.resize(img, (1200, 1600)))
cv2.waitKey(0)
