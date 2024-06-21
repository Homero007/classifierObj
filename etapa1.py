# -*- coding: utf-8 -*-
"""etapa1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jykdW-1I3ffP3dj_Ry2tDzqetA6vTtJ8
"""

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load and preprocess images
def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (224, 224))  # Resize to model expected size
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Define a simple region proposal network (RPN) for region of interest
def create_rpn(base_layers, num_anchors):
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal')(base_layers)
    x_class = Conv2D(num_anchors * 1, (1, 1), activation='sigmoid', kernel_initializer='uniform')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero')(x)
    return [x_class, x_regr]

# Main model for feature extraction
def feature_extraction_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    return feature_extractor

# Load dataset and prepare data
image_paths = ["/content/snap07071.jpg", "/content/21fdf.jpg"] # Example paths
images = np.array([load_and_preprocess_image(path) for path in image_paths])

# Feature extraction
model = feature_extraction_model()
features = model.predict(images)

# Assuming bounding box annotations and labels are available
# Annotations would typically be in the format [x_min, y_min, x_max, y_max]
annotations = [[10, 10, 100, 100], [50, 50, 200, 200]]  # Example annotations

# Use annotations to extract regions of interest
rois = []
for img, bbox in zip(images, annotations):
    x_min, y_min, x_max, y_max = bbox
    roi = img[y_min:y_max, x_min:x_max]
    rois.append(roi)

# Display the regions of interest
for roi in rois:
    cv2_imshow(roi)