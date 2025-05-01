# IMPORT LIBRARIES

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import cv2
from sklearn.metrics import confusion_matrix
from keras.models import load_model

-------------------------------------------------------------------------------------------------------------
# DATA COLLECTION AND PRE-PROCESSING

# Set the image size and data directory path
IMAGE_SIZE = 224
data_dir = ""

# Class categories (update as per your dataset)
categories = [""]

# Function to load images and labels
def load_data(data_dir):
    data = []
    labels = []

    # Load images for each category
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)  # Assign label 0, 1, 2, etc.

        for img_name in os.listdir(path):
            try:
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(path, img_name)
                    img = cv2.imread(img_path)

                if img is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))  # Resize image
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    return np.array(data), np.array(labels)

# Load the dataset
X, y = load_data(data_dir)

# Normalize the images (scale pixel values to [0, 1])
X = X.astype('float32') / 255.0

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training data
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% validation, 15% test

# Print the shapes of the splits
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Check dataset dimensions
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

----------------------------------------------------------------------------------------------------------------
# DISPLAY SAMPLE IMAGES

import random
import matplotlib.pyplot as plt
import cv2  # Assuming you're using OpenCV for image loading

plt.figure(dpi=250) 

# Display random images and their labels from the training set
for i in range(5):  # Adjust range to display more or fewer images
    idx = random.randint(0, len(X_train) - 1)  # Randomly select an image index
    img = cv2.cvtColor(X_train[idx], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color
    plt.imshow(img)
    plt.title(categories[y_train[idx]])  # Display the corresponding label
    plt.axis('off')  # Hide axes for a cleaner image display
    plt.show()

----------------------------------------------------------------------------------------------------------------
# DATA AUGMENTATION FOR BETTER GENERALIZATION

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply data augmentation to the training data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use the augmented data generator on training data
train_datagen = datagen.flow(X_train, y_train, batch_size=32)

----------------------------------------------------------------------------------------------------------------
# CNN MODEL DEFINITION

from keras.layers import Input
from keras import models, layers

model = models.Sequential([
    Input(shape=(64,64,3)),  # InputLayer'ı buraya ekleyin
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
--------------------------------------------------------------------------------------------------------------
# CNN MODEL DEFINITION VGG16

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Freeze the base model layers (so they won't be updated during training)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the pre-trained model
x = base_model.output
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Flatten()(x)
x = layers.Dense(224, activation='relu')(x)
x = layers.Dense(len(categories), activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=x)
---------------------------------------------------------------------------------------------------------------
# COMPILE THE MODEL

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
--------------------------------------------------------------------------------------------------------------
# PRINT MODEL SUMMARY

model.summary() 
--------------------------------------------------------------------------------------------------------------
# MODEL TRAINING

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
--------------------------------------------------------------------------------------------------------------
# TEST ACCURACY

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
--------------------------------------------------------------------------------------------------------------
# PLOT TRAINING AND VALIDATION ACCURACY AND LOSS

from sklearn.metrics import confusion_matrix, classification_report

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report (Precision, Recall, F1 Score)
class_report = classification_report(y_test, y_pred, target_names=categories)
print("Classification Report:\n", class_report)
--------------------------------------------------------------------------------------------------------------
# CONFUSION MATRIX

from sklearn.metrics import confusion_matrix, classification_report

# Predictions on the test set
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report (Precision, Recall, F1 Score)
class_report = classification_report(y_test, y_pred, target_names=categories)
print("Classification Report:\n", class_report)

---------------------------------------------------------------------------------------------------------------
# SAVE AND LOAD MODEL

model.save("my_model.keras")

from tensorflow.keras.optimizers import RMSprop

# Define the optimizer
optimizer = RMSprop()

# Load the saved model with the specified optimizer
model = load_model("my_model.keras", custom_objects={'RMSprop': optimizer})
--------------------------------------------------------------------------------------------------------------
# OBJECT DETECTION

def detect_and_display(frame, model, categories, image_size=224, confidence_threshold=80):
    # Resize and preprocess the frame
    resized_frame = cv2.resize(frame, (image_size, image_size))
    resized_frame = resized_frame.astype('float32') / 255.0  # Normalize

    # Expand dimensions to match the model input shape
    input_data = np.expand_dims(resized_frame, axis=0)
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # Set labelq if confidence exceeds threshold
    if confidence > confidence_threshold:
        label = f"{categories[predicted_class]}: {confidence:.2f}%"
    else:
        label = "Uncertain"

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    height, width, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 2)

    return frame

# Example of using the function with webcam
cap = cv2.VideoCapture(0)  # Open webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and display the object in the webcam frame
    frame = detect_and_display(frame, model, categories)
    
    # Show the frame
    cv2.imshow('Object Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
