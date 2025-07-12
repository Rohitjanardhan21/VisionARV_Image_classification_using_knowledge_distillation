import os
import shutil
import random
from tqdm import tqdm

def prepare_split(src_root, split):
    dst_blur = os.path.join("data/Dataset/Split", split, "blurry")
    dst_sharp = os.path.join("data/Dataset/Split", split, "sharp")
    os.makedirs(dst_blur, exist_ok=True)
    os.makedirs(dst_sharp, exist_ok=True)

    all_matched = 0

    for folder in os.listdir(src_root):
        folder_path = os.path.join(src_root, folder)
        blur_dir = os.path.join(folder_path, "blur")
        sharp_dir = os.path.join(folder_path, "sharp")

        if not os.path.isdir(blur_dir) or not os.path.isdir(sharp_dir):
            continue

        blur_files = set(os.listdir(blur_dir))
        sharp_files = set(os.listdir(sharp_dir))
        matched = sorted(blur_files & sharp_files)

        for file in tqdm(matched, desc=f"Processing {folder}"):
            shutil.copy(os.path.join(blur_dir, file), os.path.join(dst_blur, file))
            shutil.copy(os.path.join(sharp_dir, file), os.path.join(dst_sharp, file))
            all_matched += 1

        # Optional: remove GOPR folder if empty
        try:
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"Couldn't delete {folder_path}: {e}")

    print(f"✅ {all_matched} matched image pairs copied to '{split}/blurry' and '{split}/sharp'.")


def create_val_split(train_dir, val_dir, val_ratio=0.1):
    train_blur = os.path.join(train_dir, "blurry")
    train_sharp = os.path.join(train_dir, "sharp")
    val_blur = os.path.join(val_dir, "blurry")
    val_sharp = os.path.join(val_dir, "sharp")

    os.makedirs(val_blur, exist_ok=True)
    os.makedirs(val_sharp, exist_ok=True)

    all_files = sorted(os.listdir(train_blur))
    val_size = max(1, int(len(all_files) * val_ratio))
    val_files = random.sample(all_files, val_size)

    for f in val_files:
        shutil.move(os.path.join(train_blur, f), os.path.join(val_blur, f))
        shutil.move(os.path.join(train_sharp, f), os.path.join(val_sharp, f))

    print(f"✅ {val_size} image pairs moved to validation split.")


# --------- Run processing ---------
prepare_split("data/Dataset/train", "train")
prepare_split("data/Dataset/test", "test")

create_val_split("data/Dataset/Split/train", "data/Dataset/Split/val", val_ratio=0.1)
