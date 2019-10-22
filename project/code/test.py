#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import numpy as np

# Prepare the data
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
Batch_size =32
test_gen = test_datagen.flow_from_directory(
    '/dl_data/test/',
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=Batch_size,
    seed=42)

test_gen.reset()

# Load the trained model
model = load_model('model.h5')
print("Model loaded")

# Predict on test dataset
pred = model.predict_generator(test_gen, steps=test_gen.n//Batch_size+1)
pred = np.argmax(pred, axis=1)

# Calculate confusion matrix and some metrics
cm = confusion_matrix(test_gen.classes, pred)
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
recall = cm[0, 0] / (cm[0, 0] + cm[0, 1]) # sensitivity
precision = cm[0,0] / (cm[0, 0] + cm[1, 0])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
f1 = 2 * cm[0,0] / (2 * cm[0,0] + cm[1, 0] + cm[0, 1])

print(cm)
print("accuracy: {:.4f}".format(accuracy))
print("recall: {:.4f}".format(recall))
print("precision: {:.4f}".format(precision))
print("specificity: {:.4f}".format(specificity))
print("F1: {:.4f}".format(f1))

