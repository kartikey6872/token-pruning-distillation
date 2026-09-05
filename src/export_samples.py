import os
from torchvision import datasets
from PIL import Image

DATA_DIR = "./data"
OUTPUT_DIR = "./sample_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

# Save one example image per class
saved_classes = set()
for image, label in test_set:
    class_name = CIFAR10_CLASSES[label]
    if class_name not in saved_classes:
        # Upscale from 32x32 to 256x256 so it's easier to see/upload (still same content)
        image_large = image.resize((256, 256), Image.NEAREST)
        image_large.save(os.path.join(OUTPUT_DIR, f"{class_name}.png"))
        saved_classes.add(class_name)

    if len(saved_classes) == len(CIFAR10_CLASSES):
        break

print(f"Saved {len(saved_classes)} sample images to {OUTPUT_DIR}/")
print("Files:", os.listdir(OUTPUT_DIR))