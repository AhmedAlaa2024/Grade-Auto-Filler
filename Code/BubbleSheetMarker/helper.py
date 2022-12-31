import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from skimage.exposure import histogram
from matplotlib.pyplot import bar

# Show the figures / plots inside the notebook
def show_images_1(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def show_images(titles, images, wait=True):
    """Display multiple images with one line of code"""

    for (title, image) in zip(titles, images):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL) 
        cv2.imshow(title, image)

    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
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
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def skewCorrection(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged[:, -1] = np.ones(edged.shape[0])
    edged[:, 0] = np.ones(edged.shape[0])
    edged[-1, :] = np.ones(edged.shape[1])
    edged[0, :] = np.ones(edged.shape[1])

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[1:6]

    h, w = edged.shape
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            p1, p2, p3, p4 = approx
            p1, p2, p3, p4 = p1[0], p2[0], p3[0], p4[0]
            if (p1[0] < 10 and p1[1] < 10) or (w-p3[0] < 10 and h-p3[1] < 10) or (w-p2[0] < 10 and p2[1] < 10) or (p4[0] < 10 and h-p4[1] < 10):
                continue
            screenCnt = approx
            break
    img = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return img

def isFilled(image, A):
    Ax0 = A[0] - 5
    Ay0 = A[1] - 5
    Ax1 = A[0] + 5
    Ay1 = A[1] + 5
    circleImage = image[Ay0:Ay1,Ax0:Ax1]
    circleImage[circleImage < 160] = 0
    sumPixels = np.sum(circleImage)

    
    if sumPixels < 3000:
        # cv2.circle(image, (A[0], A[1]), A[2], (0, 255, 0), 2)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        return True
    else:
        return False

def extractStudentID(id_length, bubblesList):
    STUDENT_ID = [] * id_length
    bubblesList = np.array(bubblesList)
    bubblesList = np.transpose(bubblesList)
    # 10 for the number of decimal digits
    for i in range(id_length):
        if 'A' in bubblesList[i]:
            STUDENT_ID[0] = (i + 1) % 10
        if 'B' in bubblesList[i]:
            STUDENT_ID[1] = (i + 1) % 10
        if 'C' in bubblesList[i]:
            STUDENT_ID[2] = (i + 1) % 10
        if 'D' in bubblesList[i]:
            STUDENT_ID[3] = (i + 1) % 10
        if 'E' in bubblesList[i]:
            STUDENT_ID[4] = (i + 1) % 10

    s = [str(i) for i in STUDENT_ID]

    return s[0]