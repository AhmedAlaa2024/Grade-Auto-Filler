
import os
import cv2
from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib

images = []
labels = []

# get all the image folder paths
image_paths = os.listdir("../Dataset/symbols")

for path in image_paths:
	# get all the image names
	all_images = os.listdir(f"../Dataset/symbols/{path}")

	# iterate over the image names, get the label
	for image in all_images:
		image_path = f"../Dataset/symbols/{path}/{image}"
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		image = cv2.resize(image, (128, 256))

		# get the HOG descriptor for the image
		hog_desc = hog(image, orientations=9, pixels_per_cell=(8, 8),
				cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

		# update the data and labels
		images.append(hog_desc)
		labels.append(path)


# train Linear SVC 
print('Training on train images...')
svm_model = LinearSVC(random_state=42, tol=1e-5)
svm_model.fit(images, labels)

# save the model
joblib.dump(svm_model, "HOG_Model.npy")