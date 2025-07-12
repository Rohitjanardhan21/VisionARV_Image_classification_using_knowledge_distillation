import tensorflow as tf
import os

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 🔁 Decode image with correct shape + type
def decode_image(path):
    img_bytes = tf.io.read_file(path)
    file_ext = tf.strings.lower(tf.strings.split(path, ".")[-1])
    
    def decode_jpeg(): return tf.image.decode_jpeg(img_bytes, channels=3)
    def decode_png():  return tf.image.decode_png(img_bytes, channels=3)
    
    image = tf.case(
        [(tf.equal(file_ext, "jpg"), decode_jpeg),
         (tf.equal(file_ext, "jpeg"), decode_jpeg)],
        default=decode_png
    )
    image.set_shape([None, None, 3])  # ✅ Required for tf.image.resize
    return image

# 🔁 Load a blurry–sharp pair
def load_image_pair(blurry_path, sharp_path):
    blurry = decode_image(blurry_path)
    sharp  = decode_image(sharp_path)

    blurry = tf.image.resize(blurry, IMG_SIZE) / 255.0
    sharp  = tf.image.resize(sharp, IMG_SIZE) / 255.0
    return blurry, sharp

# 🔁 Create dataset from blurry/sharp dirs
def create_image_pair_dataset(blurry_dir, sharp_dir):
    blurry_paths = sorted([
        os.path.join(blurry_dir, f) for f in os.listdir(blurry_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    sharp_paths = sorted([
        os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    dataset = tf.data.Dataset.from_tensor_slices((blurry_paths, sharp_paths))
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

# 🔍 Dataset check helper
def check_dataset(dataset, name):
    count = sum(1 for _ in dataset)
    print(f"✅ {name} dataset loaded with {count} batches.")

# 📂 Set dataset base path
base_path = r"C:\Users\rohit\image_classification_and_knowledge_distillation\data\Dataset\Split"

# 📦 Load datasets
train_ds = create_image_pair_dataset(
    os.path.join(base_path, "train", "blurry"),
    os.path.join(base_path, "train", "sharp")
)

val_ds = create_image_pair_dataset(
    os.path.join(base_path, "val", "blurry"),
    os.path.join(base_path, "val", "sharp")
)

test_ds = create_image_pair_dataset(
    os.path.join(base_path, "test", "blurry"),
    os.path.join(base_path, "test", "sharp")
)

# 🧪 Verify loading
check_dataset(train_ds, "Train")
check_dataset(val_ds, "Val")
check_dataset(test_ds, "Test")

# 🖼️ Print example shape
for blurry, sharp in train_ds.take(1):
    print(f"📦 Sample batch shape → Blurry: {blurry.shape}, Sharp: {sharp.shape}")
