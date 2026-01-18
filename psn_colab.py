# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from datetime import datetime

from google.colab import drive
drive.mount('/content/drive')

dataset = '/content/drive/MyDrive/Sieci neuronowe projekt/dataset'

import os
os.listdir(dataset)

img_size = (96, 96)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size,
)

val_size = 0.5
val_batches = tf.data.experimental.cardinality(val_ds).numpy()
num_test = int(val_batches * val_size)

test_ds = val_ds.take(num_test)
val_ds = val_ds.skip(num_test)

class_names = train_ds.class_names
print("Klasy:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

model = models.Sequential([

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    layers.Conv2D(32, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    layers.Conv2D(64, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    layers.Conv2D(128, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Conv2D(256, (3,3), activation='relu', padding="same"),
    layers.Conv2D(256, (3,3), activation='relu', padding="same"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(4, activation="softmax")
])

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=5
)

early_stop = EarlyStopping(
    patience=10,
    monitor='val_loss',
    restore_best_weights=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[reduce_lr, early_stop]
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}")

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()

plt.show()

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model.save(f"emotion_cnn_{timestamp}.keras")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model.save(f"/content/drive/MyDrive/Sieci neuronowe projekt/models/emotion_cnn_{timestamp}.keras")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

print("Trwa generowanie predykcji dla zbioru testowego...")

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Przewidziana klasa (Predicted)')
plt.ylabel('Prawdziwa klasa (Actual)')
plt.title('Macierz Trafień - Analiza Błędów Modelu')
plt.show()

print("\nSzczegółowy Raport Klasyfikacji:")
print(classification_report(y_true, y_pred, target_names=class_names))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

all_images = []
all_labels = []

print("Pobieranie danych testowych...")
for img_batch, label_batch in test_ds:
    all_images.append(img_batch.numpy())
    all_labels.append(label_batch.numpy())

X_test = np.concatenate(all_images, axis=0)
y_true = np.concatenate(all_labels, axis=0)

print("Generowanie predykcji...")
y_score = model.predict(X_test)
y_pred = np.argmax(y_score, axis=1)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Macierz Trafień - Analiza Błędów Modelu')
plt.ylabel('Prawda')
plt.xlabel('Predykcja')
plt.show()

y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
n_classes = 4

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC (One-vs-Rest)')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Krzywa Precision-Recall')
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()