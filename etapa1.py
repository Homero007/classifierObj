# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Ruta a la carpeta con las imágenes
folder_path = '/content/photos'

# Obtener la lista de archivos de imagen en la carpeta
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Función para convertir imagen a escala de grises
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Función para segmentar la imagen basada en umbrales
def segment_image(image):
    gray_image = convert_to_grayscale(image)
    # Aplicar un umbral global
    _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return thresholded

# Procesar y mostrar las imágenes
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        continue
    
    segmented_image = segment_image(image)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Imagen Segmentada')
    
    plt.show()
