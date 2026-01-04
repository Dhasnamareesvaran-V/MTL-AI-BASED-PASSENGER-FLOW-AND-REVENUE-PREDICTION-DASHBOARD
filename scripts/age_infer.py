import cv2
import numpy as np
from keras.models import load_model


MODEL_PATH = r"D:\PASSENGER_FLOW_PREDICTION\models\cv\age_regressor.keras"
model = load_model(MODEL_PATH)

IMAGE_PATH = r"D:\PASSENGER_FLOW_PREDICTION\data\test\Test_Image_1.jpg"

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (128, 128))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

predicted_age = model.predict(img)[0][0]
print(f"Predicted Age: {predicted_age:.1f} years")

def age_group(age):
    if age < 13: return "Child"
    elif age < 25: return "Young Adult"
    elif age < 45: return "Adult"
    elif age < 65: return "Middle Aged"
    else: return "Senior"

print(f"Mapped Age Group: {age_group(predicted_age)}")
