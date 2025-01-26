import os           # for the Operating system
import cv2          # for OpenCV
import numpy as np
from sklearn.model_selection import train_test_split     # to utilize the library for splitting the data into training and testing sets
from keras.models import Sequential    # to define neural network layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # these are neural network layers
from keras.utils import to_categorical # use this library for one hot encoding labels

# we are going to define path directories for training images for fire & non-fire images
training_data = [
    "D:/Smoke & Fire Detection/Dataset/FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/train/Smoke",
    "D:/Smoke & Fire Detection/Dataset/FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET/train/non fire"
]

# Now we will create a function to load the images and corresponding labels
def load_images(training_data):
    # initialize an empty list for images
    images = []
    # initialize another empty list to store labels
    labels =[]
    for i in range(len(training_data)):
        folder = training_data[i]
        label = i
        for filename in os.listdir(folder):
            try:
                # read the image in grayscale and resize it to fixed dimensions
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {os.path.join(folder, filename)}: {e}")
    return np.array(images), np.array(labels)

images, labels = load_images(training_data)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255

# One hot encode labels
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Build convolutional neural network model
model = Sequential()
# Add convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) # prevent overfitting
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model with specified loss function, optimizer, and metrics
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_data=(X_test, Y_test))

# Save the model
model.save("smoke_detection_model.h5")
