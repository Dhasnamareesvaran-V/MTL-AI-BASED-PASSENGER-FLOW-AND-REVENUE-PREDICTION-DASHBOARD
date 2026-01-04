import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm   # progress bar

DATASET_PATH = r"D:\PASSENGER_FLOW_PREDICTION\images\processed"

images = []
ages = []

for file in tqdm(os.listdir(DATASET_PATH), desc="Loading images"):
    try:
        # UTKFace filename format: age_gender_race_date.jpg
        age = int(file.split("_")[0])
        img_path = os.path.join(DATASET_PATH, file)
        img = cv2.imread(img_path)

        if img is None:
            continue  # skip unreadable files

        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0

        images.append(img)
        ages.append(age)
    except Exception as e:
        continue

X = np.array(images)
y = np.array(ages)

print(f"✅ Loaded {len(X)} images with age labels")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="linear")  # regression output
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss="mae", metrics=["mae"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=16   # smaller batch size for stability
)

MODEL_PATH = r"D:\PASSENGER_FLOW_PREDICTION\models\cv\age_regressor.keras"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"🎉 Model saved at {MODEL_PATH}")
