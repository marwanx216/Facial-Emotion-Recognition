# IMPORTS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Input,
    BatchNormalization, Activation, SeparableConv2D
)
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import torch

# PARAMETERS
IMG_SIZE = 64
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 1


# DATA LOADER
def load_data(csv_path, img_folder, name, img_size=64):
    cache_file = f"{name}_cache.pt"
    if os.path.exists(cache_file):
        print(f"Loading {name} dataset from cache: {cache_file}")
        images, labels = torch.load(cache_file)
    else:
        print(f"Building {name} dataset...")
        df = pd.read_csv(csv_path)
        images, labels = [], []
        label_cols = df.columns[1:]
        for _, row in tqdm(df.iterrows(), total=len(df)):
            img_path = os.path.join(img_folder, row['filename']).replace("\\", "/")
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            images.append(img)
            label = np.argmax(row[label_cols].values)
            labels.append(label)
        images = np.array(images).reshape(-1, img_size, img_size, 1)
        labels = np.array(labels)
        np.savez_compressed(cache_file, images=images, labels=labels)
        print(f"Saved {name} cache to {cache_file}")
    if os.path.exists(cache_file) and 'images' not in locals():
        cached = np.load(cache_file)
        images, labels = cached['images'], cached['labels']
    print(f"Loaded {len(images)} images for {name}")
    print(f"Unique labels in {name} set: {np.unique(labels)}")
    return images, labels


# LOAD DATA
train_csv = "train/train_labels.csv"
val_csv = "valid/valid_labels.csv"
test_csv = "test/_test_labels.csv"
train_img_folder = "train"
val_img_folder = "valid"
test_img_folder = "test"

X_train, y_train = load_data(train_csv, train_img_folder, "train_cache", IMG_SIZE)
X_val, y_val = load_data(val_csv, val_img_folder, "val_cache", IMG_SIZE)
X_test, y_test = load_data(test_csv, test_img_folder, "test_cache", IMG_SIZE)

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")

y_train_cat = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_cat = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


# VISION TRANSFORMER (ViT)
def build_vit(img_size=64, num_classes=7):
    data_augmentation = keras.Sequential([
        layers.Resizing(img_size, img_size),
        layers.Rescaling(1. / 255),
    ])
    inputs = layers.Input(shape=(img_size, img_size, 1))
    x = layers.Conv2D(3, (3, 3), padding="same")(inputs)
    x = data_augmentation(x)
    patch_size = 8
    num_patches = (img_size // patch_size) ** 2
    projection_dim = 64
    patch_encoder = keras.Sequential([
        layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid"),
        layers.Reshape((-1, projection_dim)),
    ])
    x = patch_encoder(x)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_embed
    for _ in range(6):
        x1 = layers.LayerNormalization()(x)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([x, attention_output])
        x3 = layers.LayerNormalization()(x2)
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim, activation="gelu")(x3)
        x = layers.Add()([x2, x3])
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)


# CNN MODEL
def build_cnn(img_size=64, num_classes=7):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(img_size, img_size, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ]) 
    return model


# TRAIN AND EVALUATE FUNCTION
def train_and_evaluate(model, model_name, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5,weight_decay=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-7, mode="max"),
        tf.keras.callbacks.ModelCheckpoint(f"best_{model_name}_model.h5", monitor="val_accuracy", save_best_only=True,
                                           mode="max")
    ]
    print(f"\nTraining {model_name} model...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    print(f"\n{model_name} Model Evaluation:")
    train_loss, train_acc = model.evaluate(X_train, y_train_cat, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_labels,
                yticklabels=emotion_labels)
    plt.title(f'{model_name} Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=emotion_labels))

    return history, test_acc


# BUILD, TRAIN AND EVALUATE MODELS
cnn_model = build_cnn(IMG_SIZE, NUM_CLASSES)
vit_model = build_vit(IMG_SIZE, NUM_CLASSES)

print("\nCNN Model Summary:")
cnn_model.summary()

print("\nVision Transformer (ViT) Model Summary:")
vit_model.summary()

cnn_history, cnn_test_acc = train_and_evaluate(
    cnn_model, "CNN",
    X_train, y_train_cat,
    X_val, y_val_cat,
    X_test, y_test_cat,
    y_test
)

vit_history, vit_test_acc = train_and_evaluate(
    vit_model, "ViT",
    X_train, y_train_cat,
    X_val, y_val_cat,
    X_test, y_test_cat,
    y_test
)



# MODEL COMPARISON
print("\nModel Comparison:")
print(f"ViT Test Accuracy: {vit_test_acc:.4f}")
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")

if vit_test_acc > cnn_test_acc:
    print("ViT performed better than CNN")
elif cnn_test_acc > vit_test_acc:
    print("CNN performed better than ViT")
else:
    print("Both models performed equally")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(vit_history.history['accuracy'], label='ViT Train')
plt.plot(vit_history.history['val_accuracy'], label='ViT Val')
plt.plot(cnn_history.history['accuracy'], label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Val')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(vit_history.history['loss'], label='ViT Train')
plt.plot(vit_history.history['val_loss'], label='ViT Val')
plt.plot(cnn_history.history['loss'], label='CNN Train')
plt.plot(cnn_history.history['val_loss'], label='CNN Val')
plt.title('Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()


