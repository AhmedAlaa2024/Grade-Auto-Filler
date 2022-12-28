import cv2
import joblib
from skimage.feature import hog

HOG = joblib.load("HOG_Model.npy")

image = cv2.imread(f"../Dataset/symbols/vertical5/Cell_0_Test_5.jpg", cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (128, 256))

# get the HOG descriptor for the test image
(hog_desc, hog_image) = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys', visualize=True)

# prediction
pred = HOG.predict(hog_desc.reshape(1, -1))[0]
print(pred.title())