import tensorflow
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Grab the labels from the labels.txt file. This will be used later.
labels = open('labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = cv2.imread("../Dataset/question/Cell_0_Test_10.jpg")

image = cv2.resize(image, (224, 224))
image = np.asarray(image)
# Normalize the image
normalizedImage = (image.astype(np.float32) / 127.0) - 1

# Load the image into the array
TM_DATA[0] = normalizedImage
PredictionVar = model.predict(TM_DATA)

print(labels[np.argmax(PredictionVar)])

