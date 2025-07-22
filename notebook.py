#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Paths
train_dir = "/kaggle/input/plantvillage/PlantVillage/train"
val_dir = "/kaggle/input/plantvillage/PlantVillage/val"  # Final test set

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
INIT_LR = 1e-4

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # 10% of training for validation
)

# Training generator
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Final test generator (never used in training)
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Class weights
train_classes = train_gen.classes
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weights = dict(enumerate(class_weights))

# Print class distribution
print("Class distribution:")
for class_idx, count in enumerate(np.bincount(train_classes)):
    class_name = list(train_gen.class_indices.keys())[class_idx]
    print(f"{class_name}: {count} samples (weight: {class_weights[class_idx]:.2f})")

# Model
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMAGE_SIZE, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint('densenet_best.h5', monitor='val_auc', save_best_only=True)
]

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# Plot results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()

# Save model
model.save("plant_disease_densenet.h5")
print("Training complete! Best model saved as 'densenet_best.h5'")

# Final evaluation on completely unseen test set
print("\nFinal evaluation on test set:")
test_metrics = model.evaluate(test_gen)
for name, value in zip(model.metrics_names, test_metrics):
    print(f"{name}: {value:.4f}")

# Sample predictions from test set
sample_batch = next(test_gen)
predictions = model.predict(sample_batch[0])
class_names = list(train_gen.class_indices.keys())

print("\nSample predictions:")
for i in range(5):
    pred_class = np.argmax(predictions[i])
    true_class = np.argmax(sample_batch[1][i])
    print(f"Image {i+1}:")
    print(f"  Predicted: {class_names[pred_class]} ({predictions[i][pred_class]:.2%})")
    print(f"  Actual:    {class_names[true_class]}")
    print("---")


# In[9]:


import os
from collections import defaultdict

def count_images(directory):
    class_counts = defaultdict(int)
    total = 0
    for class_name in os.listdir(directory):
        class_folder = os.path.join(directory, class_name)
        if os.path.isdir(class_folder):
            count = len([
                f for f in os.listdir(class_folder)
                if os.path.isfile(os.path.join(class_folder, f))
            ])
            class_counts[class_name] = count
            total += count
    return class_counts, total

train_dir = "/kaggle/input/plantvillage/PlantVillage/train"
val_dir = "/kaggle/input/plantvillage/PlantVillage/val"

train_counts, total_train = count_images(train_dir)
val_counts, total_val = count_images(val_dir)

print("🔹 Train Set Counts:")
for cls, cnt in train_counts.items():
    print(f"{cls}: {cnt} images")
print(f"\n✅ Total Train Images: {total_train}")

print("\n🔹 Validation Set Counts:")
for cls, cnt in val_counts.items():
    print(f"{cls}: {cnt} images")
print(f"\n✅ Total Validation Images: {total_val}")


# In[11]:


model.save("plant_disease_densenet.keras")


# In[ ]:


from tensorflow.keras.models import load_model

model = load_model("/kaggle/input/plant-disease-detector/keras/default/1/plant_disease_densenet.keras")


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    '/kaggle/input/plantvillage/PlantVillage/val',  # ✅ Replace this with your actual test folder
    target_size=(224, 224),    # ✅ Use the input size your model expects
    batch_size=32,
    class_mode='categorical',
    shuffle=False              # ✅ Important: do NOT shuffle so labels stay in order
)




# ✅ Predict on the entire test set
y_test_probs = model.predict(test_gen, verbose=1)
y_test_preds = np.argmax(y_test_probs, axis=1)

# ✅ Extract true labels (one-hot to class index)
y_test_true = np.argmax(test_gen.labels, axis=-1) if test_gen.labels.ndim == 2 else test_gen.labels

# ✅ Get class names
class_names = list(test_gen.class_indices.keys())

# ✅ Classification Report
report = classification_report(y_test_true, y_test_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("final_test_classification_report.csv", index=True)
print("\n✅ Classification Report Saved: final_test_classification_report.csv")

# ✅ Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_preds)
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(include_values=True, cmap="Blues", xticks_rotation=90, ax=ax)
plt.title("Confusion Matrix - Final Test Set")
plt.tight_layout()
plt.savefig("final_test_confusion_matrix.png")
plt.show()
print("✅ Confusion Matrix Saved: final_test_confusion_matrix.png")


# In[ ]:




