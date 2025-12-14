import os
import random
import shutil

# paths
IMAGE_DIR = "Augmented2/images"
LABEL_DIR = "Augmented2/labels"

OUT_DIR = "dataset"

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# create output directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

# get image files
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

def copy_files(image_list, split):
    for img in image_list:
        img_path = os.path.join(IMAGE_DIR, img)
        label_path = os.path.join(LABEL_DIR, img.replace(".jpg", ".txt")
                                                 .replace(".png", ".txt")
                                                 .replace(".jpeg", ".txt"))

        # copy image
        shutil.copy(img_path, os.path.join(OUT_DIR, "images", split, img))

        # copy label
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(OUT_DIR, "labels", split,
                                                 os.path.basename(label_path)))
        else:
            print(f"⚠️ Label missing for {img}")

copy_files(train_images, "train")
copy_files(val_images, "val")

print("✅ Dataset split completed")
print(f"Train images: {len(train_images)}")
print(f"Val images: {len(val_images)}")
