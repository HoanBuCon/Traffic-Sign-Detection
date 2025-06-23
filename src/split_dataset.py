import os
import random
import shutil

# Đường dẫn gốc (trong thư mục hiện tại)
IMAGE_DIR = os.path.abspath("./dataset/images")
LABEL_DIR = os.path.abspath("./dataset/labels")

# Tạo thư mục mới để chứa train/val trong cùng thư mục dataset
OUTPUT_IMAGE_DIR = os.path.join(os.path.dirname(IMAGE_DIR), "images")
OUTPUT_LABEL_DIR = os.path.join(os.path.dirname(LABEL_DIR), "labels")

# Tạo thư mục con: train và val
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_LABEL_DIR, split), exist_ok=True)

# Tỷ lệ train/val
train_ratio = 0.8

# Lấy danh sách file ảnh
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
random.shuffle(image_files)

# Chia dữ liệu
train_size = int(len(image_files) * train_ratio)
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# Hàm di chuyển ảnh và nhãn
def move_files(file_list, split):
    for img in file_list:
        base = os.path.splitext(img)[0]
        label = base + ".txt"

        # Đường dẫn gốc
        img_path = os.path.join(IMAGE_DIR, img)
        label_path = os.path.join(LABEL_DIR, label)

        # Đường dẫn đích
        img_dest = os.path.join(OUTPUT_IMAGE_DIR, split, img)
        label_dest = os.path.join(OUTPUT_LABEL_DIR, split, label)

        # Chỉ di chuyển nếu cả ảnh và nhãn đều tồn tại
        if os.path.exists(img_path) and os.path.exists(label_path):
            shutil.move(img_path, img_dest)
            shutil.move(label_path, label_dest)
        else:
            print(f"⚠️ Thiếu file: {img} hoặc {label}")

move_files(train_files, "train")
move_files(val_files, "val")

print("✅ Đã chia xong dữ liệu thành train và val.")
