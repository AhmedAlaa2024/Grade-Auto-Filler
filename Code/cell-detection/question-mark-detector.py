import cv2
import numpy as np

img = cv2.imread("../Samples/Detection phase samples/question mark1.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Apply Hough transform on the blurred image. 
detected_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 80, param1 = 20, param2 = 9, minRadius = 10, maxRadius = 20) 
print(len(detected_circles))
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles)) 

    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
         
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)          # circle
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)          # center
 
else:
    print("Circle is not found")

cv2.imshow("out", img)
cv2.waitKey()