import os
import json
import numpy as np
from datetime import datetime

import keras
from keras import layers, models, callbacks, utils

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CV_DIR = os.path.join(ROOT, "data", "cv")
MODELS_CV_DIR = os.path.join(ROOT, "models", "cv")
REPORTS_CV_DIR = os.path.join(ROOT, "reports", "cv")

os.makedirs(MODELS_CV_DIR, exist_ok=True)
os.makedirs(REPORTS_CV_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_CV_DIR, "X.npy")
Y_AGE_PATH = os.path.join(DATA_CV_DIR, "y_age.npy")
Y_GENDER_PATH = os.path.join(DATA_CV_DIR, "y_gender.npy")

IMG_SIZE = (128, 128, 3)
AGE_BUCKETS = [(0, 12), (13, 24), (25, 44), (45, 64), (65, 120)]  # child, teen/young adult, adult, middle-aged, senior
AGE_CLASS_NAMES = ["child", "young_adult", "adult", "middle_aged", "senior"]
BATCH_SIZE = 32
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42

def bucketize_age(age):
    for i, (lo, hi) in enumerate(AGE_BUCKETS):
        if lo <= age <= hi:
            return i
    return len(AGE_BUCKETS) - 1

def train_val_split(X, y_age_cls, y_gender, val_ratio=VAL_SPLIT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    return (
        X[train_idx], y_age_cls[train_idx], y_gender[train_idx],
        X[val_idx], y_age_cls[val_idx], y_gender[val_idx]
    )

def build_model(input_shape=IMG_SIZE, age_classes=len(AGE_CLASS_NAMES)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    
    age_logits = layers.Dense(age_classes, activation="softmax", name="age")(x)

    
    gender_logits = layers.Dense(1, activation="sigmoid", name="gender")(x)

    model = models.Model(inputs=inputs, outputs=[age_logits, gender_logits])
    model.compile(
        optimizer="adam",
        loss={
            "age": "sparse_categorical_crossentropy",
            "gender": "binary_crossentropy",
        },
        metrics={
            "age": ["accuracy"],
            "gender": ["accuracy"],
        }
    )
    return model

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def main():
    X = np.load(X_PATH)  
    y_age = np.load(Y_AGE_PATH)  
    y_gender = np.load(Y_GENDER_PATH)  

    
    y_age_cls = np.array([bucketize_age(a) for a in y_age], dtype=np.int32)

    
    X_train, y_age_train, y_gender_train, X_val, y_age_val, y_gender_val = train_val_split(X, y_age_cls, y_gender)

    
    model = build_model()

    
    ckpt_path = os.path.join(MODELS_CV_DIR, "age_gender_cnn.keras")
    cbs = [
        callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_age_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_age_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_age_accuracy",
            mode="max",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    
    history = model.fit(
        X_train,
        {"age": y_age_train, "gender": y_gender_train},
        validation_data=(X_val, {"age": y_age_val, "gender": y_gender_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    
    eval_results = model.evaluate(X_val, {"age": y_age_val, "gender": y_gender_val}, verbose=0)
    metrics = {
        "val_age_loss": float(eval_results[1]),
        "val_gender_loss": float(eval_results[2]),
        "val_age_accuracy": float(eval_results[3]),
        "val_gender_accuracy": float(eval_results[4]),
        "samples_train": int(len(X_train)),
        "samples_val": int(len(X_val)),
        "age_classes": AGE_CLASS_NAMES,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    
    age_probs, gender_probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    age_pred = np.argmax(age_probs, axis=1)
    gender_pred = (gender_probs.flatten() >= 0.5).astype(np.int32)

    cm_age = confusion_matrix(y_age_val, age_pred, num_classes=len(AGE_CLASS_NAMES))
    cm_gender = confusion_matrix(y_gender_val.astype(int), gender_pred.astype(int), num_classes=2)

    
    metrics_path = os.path.join(REPORTS_CV_DIR, "cv_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    
    np.save(os.path.join(REPORTS_CV_DIR, "cm_age.npy"), cm_age)
    np.save(os.path.join(REPORTS_CV_DIR, "cm_gender.npy"), cm_gender)

    print(f"[INFO] Training complete.")
    print(f"[INFO] Best model saved to: {ckpt_path}")
    print(f"[INFO] Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()
