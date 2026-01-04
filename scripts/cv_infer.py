import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


MODEL_PATH = r"D:\PASSENGER_FLOW_PREDICTION\models\cv\age_gender_cnn.keras"

AGE_CLASS_NAMES = ["child", "young_adult", "adult", "middle_aged", "senior"]

def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(img_path):
    model = load_model(MODEL_PATH)
    img_array = preprocess_image(img_path)

    age_probs, gender_probs = model.predict(img_array, verbose=0)

    age_class = np.argmax(age_probs[0])
    gender_class = int(gender_probs[0][0] >= 0.5)

    print(f"Image: {img_path}")
    print(f"Predicted Age Group: {AGE_CLASS_NAMES[age_class]}")
    print(f"Predicted Gender: {'Male' if gender_class == 1 else 'Female'}")

if __name__ == "__main__":

    test_img = r"D:\PASSENGER_FLOW_PREDICTION\data\test\Test_Image_1.jpg"
    predict(test_img)
