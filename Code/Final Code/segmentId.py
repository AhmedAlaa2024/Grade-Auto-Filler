import cv2
import numpy as np
from getPredection import getPrediction


def segmentId(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 * img
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    res, img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours((img).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr))
    white_img_large_contours = np.ones(img.shape)
    dimensions_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (w * h > 40):
            dimensions_contours.append((x, y, w, h))
            cv2.rectangle(white_img_large_contours, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cropped_digits = []
    i = 0
    filtered_img = img
    for dimension in dimensions_contours:
        (x, y, w, h) = dimension
        imgCopy = filtered_img[y - 1:y + h + 1, x - 1:x + w + 1]
        x = imgCopy.shape[1]
        number_of_images = 1
        if (x > 25):
            number_of_images = np.ceil(x / 23.0)
            for j in range(int(number_of_images)):
                imgTemp = cv2.resize(imgCopy[:, j * 23:(j + 1) * 23], (200, 100))
                imgTemp = cv2.resize(imgTemp, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                cropped_digits.append(imgTemp)
                i += 1
        else:
            if imgCopy.size == 0:
                continue
            imgCopy = cv2.resize(imgCopy, (200, 100))
            cropped_digits.append(imgCopy)
            imgCopy = cv2.resize(imgCopy, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            i += 1

    return cropped_digits


# A function used to extract the id from image
# it takes the image then segment the digits from it and get the prediction from each image
def getIdFromImage(img):
    cropped_digits = segmentId(img)
    predictedNumber = ""

    for img in cropped_digits:
        # print(getPrediction(img))
        predictedNumber += str(getPrediction(img))
    return predictedNumber
