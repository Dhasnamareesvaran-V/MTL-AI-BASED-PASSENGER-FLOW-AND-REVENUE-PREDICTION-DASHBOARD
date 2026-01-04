import os
import numpy as np
from PIL import Image

src = r"D:\PASSENGER_FLOW_PREDICTION\images\processed"
dst = r"D:\PASSENGER_FLOW_PREDICTION\data\cv"
os.makedirs(dst, exist_ok=True)

IMG_SIZE = (128, 128)  

X = []
y_age = []
y_gender = []

def parse_labels(filename):
    """
    UTKFace filename format: age_gender_race_date&time.jpg.chip.jpg
    Example: 25_0_0_20170116174525125.jpg.chip.jpg
    """
    parts = filename.split("_")
    try:
        age = int(parts[0])
        gender = int(parts[1])  
        return age, gender
    except Exception:
        return None, None

for fname in os.listdir(src):
    if fname.lower().endswith(".jpg"):
        fpath = os.path.join(src, fname)
        age, gender = parse_labels(fname)
        if age is None or gender is None:
            continue

        img = Image.open(fpath).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img) / 255.0  

        X.append(arr)
        y_age.append(age)
        y_gender.append(gender)

X = np.array(X, dtype=np.float32)
y_age = np.array(y_age, dtype=np.int32)
y_gender = np.array(y_gender, dtype=np.int32)

np.save(os.path.join(dst, "X.npy"), X)
np.save(os.path.join(dst, "y_age.npy"), y_age)
np.save(os.path.join(dst, "y_gender.npy"), y_gender)

print(f"Preprocessing complete!")
print(f"Images shape: {X.shape}")
print(f"Ages shape: {y_age.shape}, Genders shape: {y_gender.shape}")
print(f"Saved to: {dst}")
