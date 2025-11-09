import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

IMG_SIZE = 64
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def load_test_data(csv_path, img_folder, img_size=64):
    df = pd.read_csv(csv_path)
    images, filenames = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_folder, row['filename']).replace("\\", "/")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        images.append(img)
        filenames.append(row['filename'])
    images = np.array(images).reshape(-1, img_size, img_size, 1)
    return images, filenames


model_path = "best_CNN_model.h5"
model = keras.models.load_model(model_path)

test_csv = "exam_test/exam_test.csv"
test_img_folder = "exam_test"
X_test, filenames = load_test_data(test_csv, test_img_folder, IMG_SIZE)

y_pred = np.argmax(model.predict(X_test), axis=1)

output = pd.DataFrame({
    "filename": filenames,
    "predicted_label": [emotion_labels[i] for i in y_pred]
})
output.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")

