import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# =======================
# CONFIGURATION
# =======================
# Folder containing your already predicted images
EXISTING_OUTPUT_DIR = "runs/detect/predict"   # Change to segment folder if needed
EXTENSIONS = ['jpg', 'jpeg', 'png']
SHOW_EXAMPLES = True
CSV_SUMMARY = "runs/analyze/existing_prediction_summary.csv"
CONF_HIST = "runs/analyze/existing_confidence_histogram.png"

# =======================
# PREPARE OUTPUT FOLDER
# =======================
os.makedirs(os.path.dirname(CSV_SUMMARY), exist_ok=True)

# =======================
# COLLECT IMAGE PATHS
# =======================
image_paths = []
for ext in EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(EXISTING_OUTPUT_DIR, f"*.{ext}")))

if not image_paths:
    raise FileNotFoundError(f"No images found in '{EXISTING_OUTPUT_DIR}' with extensions {EXTENSIONS}")

print(f"Found {len(image_paths)} predicted images for analysis.")

# =======================
# ANALYSIS VARIABLES
# =======================
all_classes = []
all_confidences = []

# =======================
# EXTRACT CLASS AND CONFIDENCE FROM FILE NAMES (YOLO usually embeds in filename)
# If you don’t have info, we can only do qualitative + count images
# =======================
for img_path in image_paths:
    # For simplicity, count one object per image (you can modify if you have txt labels)
    all_classes.append("object_detected")  # placeholder
    # No confidence info in saved images, skip confidences

print(f"\nTotal images analyzed: {len(all_classes)}")

# =======================
# CLASS COUNT
# =======================
class_count = Counter(all_classes)
print("\n📊 Class-wise Object Count (approx):")
for cls, num in class_count.items():
    print(f"{cls}: {num}")

# =======================
# SHOW EXAMPLE IMAGE(S)
# =======================
if SHOW_EXAMPLES:
    print("\n🖼️ Showing first example image...")
    img = Image.open(image_paths[0])
    img.show()

# =======================
# SAVE CSV SUMMARY
# =======================
summary_df = pd.DataFrame({
    "class": all_classes,
    "image": [os.path.basename(p) for p in image_paths],
})
summary_df.to_csv(CSV_SUMMARY, index=False)
print(f"\n💾 CSV summary saved to {CSV_SUMMARY}")
