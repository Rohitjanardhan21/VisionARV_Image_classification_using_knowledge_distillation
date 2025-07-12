import os
import cv2
import shutil

def move_images_to_sharp(base_dir):
    # Go through each split
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        sharp_dir = os.path.join(split_dir, 'sharp')
        os.makedirs(sharp_dir, exist_ok=True)

        for img_name in os.listdir(split_dir):
            img_path = os.path.join(split_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                shutil.move(img_path, os.path.join(sharp_dir, img_name))

def create_blurry_images(input_dir, output_dir, scale=6):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        # Downscale + Upscale to create blur
        downscaled = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
        # Optional Gaussian blur
        blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
        # Save
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, blurred)

    print(f"✅ Blurry images generated in {output_dir}")

# Define base directory
base_dir = r"C:\Users\rohit\image_classification_and_knowledge_distillation\data\Dataset\Split"

# Step 1: Move original sharp images into sharp/ folders
move_images_to_sharp(base_dir)

# Step 2: Create blurry versions from sharp/
for split in ['train', 'val', 'test']:
    sharp_path = os.path.join(base_dir, split, 'sharp')
    blurry_path = os.path.join(base_dir, split, 'blurry')
    create_blurry_images(sharp_path, blurry_path, scale=6)  # You can adjust scale here
