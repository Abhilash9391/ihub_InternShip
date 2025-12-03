import os
import cv2
import glob
import albumentations as A

# FOLDERS
images_path = "SignLanguage2"       # your images folder
labels_path = "BoundingBoxes"  # your YOLO labels folder

aug_images_out = "augmented/images"
aug_labels_out = "augmented/labels"

os.makedirs(aug_images_out, exist_ok=True)
os.makedirs(aug_labels_out, exist_ok=True)

# AUGMENTATION PIPELINE
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.1, rotate_limit=10, p=0.7),
    A.Blur(blur_limit=3, p=0.3),
    A.HueSaturationValue(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

AUG_PER_IMAGE = 5

for img_path in glob.glob(os.path.join(images_path, "*.png")):
    filename = os.path.basename(img_path)
    label_file = filename.replace(".png", ".txt")
    lbl_path = os.path.join(labels_path, label_file)

    if not os.path.exists(lbl_path):
        print("Missing label:", lbl_path)
        continue

    image = cv2.imread(img_path)

    bboxes = []
    class_labels = []
    with open(lbl_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            bboxes.append([xc, yc, bw, bh])
            class_labels.append(int(cls))

    for i in range(AUG_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        new_img_name = filename.replace(".png", f"_aug{i}.png")
        new_label_name = new_img_name.replace(".png", ".txt")

        cv2.imwrite(os.path.join(aug_images_out, new_img_name), aug_img)

        with open(os.path.join(aug_labels_out, new_label_name), "w") as f:
            for cls, (xc, yc, bw, bh) in zip(aug_labels, aug_bboxes):
                f.write(f"{cls} {xc} {yc} {bw} {bh}\n")

print("Augmentation finished!")
