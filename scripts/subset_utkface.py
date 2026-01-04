import os, shutil, random

src_root = r"D:\PASSENGER_FLOW_PREDICTION\images\raw"

dst = r"D:\PASSENGER_FLOW_PREDICTION\images\processed"
subset_size = 2000
random_seed = 42  # reproducible sampling

os.makedirs(dst, exist_ok=True)

all_images = []
for root, dirs, files in os.walk(src_root):
    for fname in files:
        if fname.lower().endswith(".jpg"):
            all_images.append(os.path.join(root, fname))

print(f"Found {len(all_images)} images in raw folder.")

random.seed(random_seed)
subset_paths = random.sample(all_images, subset_size)

for src_path in subset_paths:
    base = os.path.basename(src_path)
    dst_path = os.path.join(dst, base)
   
    if os.path.exists(dst_path):
        name, ext = os.path.splitext(base)
        i = 1
        while True:
            alt = f"{name}__dup{i}{ext}"
            alt_path = os.path.join(dst, alt)
            if not os.path.exists(alt_path):
                dst_path = alt_path
                break
            i += 1
    shutil.copy2(src_path, dst_path)

print(f"Subset created with {subset_size} images in {dst}")
