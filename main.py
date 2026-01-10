from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Model
model_name = "florian-morel22/segformer-b0-deepglobe-land-cover"
# Use base SegFormer processor (the fine-tuned model doesn't include preprocessor config)
processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# 2. Load Image (RGB Satellite Tile)
image = Image.open("satellite_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# 3. Predict
with torch.no_grad():
    outputs = model(**inputs)

# 4. Process Output
logits = outputs.logits
# Resize prediction to match original image size
upsampled_logits = torch.nn.functional.interpolate(
    logits, size=image.size[::-1], mode="bilinear", align_corners=False
)
# Get the class ID for every pixel (0-6)
pred_mask = upsampled_logits.argmax(dim=1)[0].numpy()

# 5. VISUALIZATION (Map IDs to Colors)
# Color Palette: Urban=Cyan, Agri=Yellow, Range=Magenta, Forest=Green, Water=Blue, Barren=White
palette = np.array([
    [0, 255, 255],   # 0: Urban
    [255, 255, 0],   # 1: Agriculture
    [255, 0, 255],   # 2: Rangeland
    [0, 255, 0],     # 3: Forest
    [0, 0, 255],     # 4: Water
    [255, 255, 255], # 5: Barren
    [0, 0, 0]        # 6: Unknown
])

# Class names for legend
class_names = ["Urban", "Agriculture", "Rangeland", "Forest", "Water", "Barren", "Unknown"]

# Create an RGB image from the mask
color_mask = palette[pred_mask]

# Create legend patches
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=np.array(palette[i])/255, label=class_names[i]) for i in range(len(class_names))]

# Plot with legend
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(color_mask)
ax.set_title("Land Cover Segmentation")
ax.axis("off")
ax.legend(handles=legend_patches, loc="upper right", title="Land Cover Classes")
plt.tight_layout()
plt.show()