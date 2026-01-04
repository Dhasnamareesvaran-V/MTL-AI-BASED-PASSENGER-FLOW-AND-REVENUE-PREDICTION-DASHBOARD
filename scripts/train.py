import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_DIR = r"D:\PASSENGER_FLOW_PREDICTION\data\cv"
MODEL_DIR = r"D:\PASSENGER_FLOW_PREDICTION\models\vision"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X.npy"))        # shape: (2000, 128, 128, 3), normalized [0,1]
y_age = np.load(os.path.join(DATA_DIR, "y_age.npy"))       # shape: (2000,)
y_gender = np.load(os.path.join(DATA_DIR, "y_gender.npy")) # shape: (2000,)

age_mean, age_std = y_age.mean(), y_age.std() if y_age.std() > 0 else 1.0
y_age_norm = (y_age - age_mean) / age_std

# Train/val/test split
X_train, X_temp, y_age_train, y_age_temp, y_gender_train, y_gender_temp = train_test_split(
    X, y_age_norm, y_gender, test_size=0.3, random_state=42, stratify=y_gender
)
X_val, X_test, y_age_val, y_age_test, y_gender_val, y_gender_test = train_test_split(
    X_temp, y_age_temp, y_gender_temp, test_size=0.5, random_state=42, stratify=y_gender_temp
)

# Model: CNN backbone + two heads
def build_model(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    shared = layers.Dense(256, activation="relu")(x)

    # Gender head (binary classification)
    gender_logits = layers.Dense(1, activation="sigmoid", name="gender")(shared)

    # Age head (regression on normalized age)
    age_out = layers.Dense(1, activation="linear", name="age")(shared)

    model = models.Model(inputs=inputs, outputs=[gender_logits, age_out])
    return model

model = build_model()

# Losses and metrics
losses = {
    "gender": "binary_crossentropy",
    "age": "mae"  # on normalized age; MAE corresponds to z-score error
}
metrics = {
    "gender": ["accuracy"],
    "age": ["mae"]
}
loss_weights = {"gender": 1.0, "age": 0.5}  # emphasize gender slightly

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=losses,
              loss_weights=loss_weights,
              metrics=metrics)

model.summary()

checkpoint_path = os.path.join(MODEL_DIR, "utkface_multitask_best.keras")
cbs = [
    callbacks.ModelCheckpoint(
        checkpoint_path, monitor="val_gender_accuracy", mode="max",
        save_best_only=True, save_weights_only=False
    ),
    callbacks.EarlyStopping(
        monitor="val_gender_accuracy", mode="max",
        patience=8, restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_gender_accuracy", mode="max",
        factor=0.5, patience=4, min_lr=1e-5
    )
]

# Train
hist = model.fit(
    X_train,
    {"gender": y_gender_train, "age": y_age_train},
    validation_data=(X_val, {"gender": y_gender_val, "age": y_age_val}),
    epochs=40,
    batch_size=32,
    callbacks=cbs,
    verbose=1
)

# Evaluate on test set
test_metrics = model.evaluate(
    X_test,
    {"gender": y_gender_test, "age": y_age_test},
    verbose=1
)
print("Test metrics (order = gender loss, age loss, gender acc, age mae):")
print(test_metrics)

# Save final model
final_model_path = os.path.join(MODEL_DIR, "utkface_multitask_final.keras")
model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")

pred_gender, pred_age_norm = model.predict(X_test, verbose=0)

pred_gender_cls = (pred_gender.flatten() >= 0.5).astype(int)

pred_age_years = (pred_age_norm.flatten() * age_std) + age_mean
true_age_years = (y_age_test * age_std) + age_mean

age_mae_years = np.mean(np.abs(pred_age_years - true_age_years))
gender_acc = (pred_gender_cls == y_gender_test).mean()

print(f"Final test — Gender accuracy: {gender_acc:.3f}, Age MAE: {age_mae_years:.2f} years")
