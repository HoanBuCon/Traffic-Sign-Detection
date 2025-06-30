import os
import json
import shutil
from tqdm import tqdm
import yaml

# Đường dẫn
INCREASE_DATASET = 'increase_dataset'
DATASET_IMAGES = 'datasetv2/images'
DATASET_LABELS = 'datasetv2/labels'
LOG_FILE = 'increase_dataset_merged.log'

# Danh sách class mới (từ bạn cung cấp)
new_classes = [
    "DP.135", "P.102", "P.103a", "P.103b", "P.103c", "P.104", "P.106a", "P.106b", "P.107a", "P.112", "P.115", "P.117", "P.123a", "P.123b", "P.124a", "P.124b", "P.124c", "P.125", "P.127", "P.128", "P.130", "P.131a", "P.137", "P.245a", "R.301a", "R.301c", "R.301d", "R.301e", "R.302a", "R.302b", "R.303", "R.407a", "R.409", "R.425", "R.434", "S.509a", "W.201a", "W.201b", "W.202a", "W.202b", "W.203b", "W.203c", "W.205a", "W.205b", "W.205d", "W.207a", "W.207b", "W.207c", "W.208", "W.209", "W.210", "W.219", "W.221b", "W.224", "W.225", "W.227", "W.233", "W.235", "W.245a"
]

# Đọc class cũ từ data.yaml
with open('data.yaml', 'r', encoding='utf-8') as f:
    data_yaml = yaml.safe_load(f)
old_classes = data_yaml['names']

# Gộp class, loại trùng, giữ thứ tự cũ trước, mới sau
all_classes = old_classes + [c for c in new_classes if c not in old_classes]

# Mapping tên class sang index
class2idx = {c: i for i, c in enumerate(all_classes)}

# Cập nhật lại data.yaml
data_yaml['names'] = all_classes
data_yaml['nc'] = len(all_classes)
with open('data.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

# Nếu đã merge rồi thì bỏ qua
if os.path.exists(LOG_FILE):
    print("increase_dataset đã được merge trước đó. Nếu muốn merge lại, hãy xóa file increase_dataset_merged.log")
    exit(0)

# Hàm chuyển bbox COCO sang YOLO
def coco2yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_c, y_c, w, h

print('Đang kiểm tra các file annotation:')
found_any = False
for split in ['train', 'valid', 'test']:
    # Sửa lại đường dẫn annotation đúng thực tế
    json_path = os.path.join(INCREASE_DATASET, split, '_annotations.coco.json')
    print(f'Kiểm tra file: {json_path} - Tồn tại: {os.path.exists(json_path)}')
    if not os.path.exists(json_path):
        print(f'Không tìm thấy file annotation cho split {split}, bỏ qua.')
        continue
    found_any = True
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {img['id']: img for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}
    # Tạo thư mục đích
    img_dst = os.path.join(DATASET_IMAGES, split)
    lbl_dst = os.path.join(DATASET_LABELS, split)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    # Gộp annotation
    imgid2anns = {}
    for ann in coco['annotations']:
        imgid2anns.setdefault(ann['image_id'], []).append(ann)
    for img_id, img in tqdm(images.items(), desc=f'Processing {split}'):
        file_name = img['file_name']
        src_img_path = os.path.join(INCREASE_DATASET, split, file_name)
        dst_img_path = os.path.join(img_dst, file_name)
        # Copy ảnh (ép copy đè)
        try:
            shutil.copy2(src_img_path, dst_img_path)
        except Exception as e:
            print(f'Lỗi copy ảnh {src_img_path} -> {dst_img_path}: {e}')
        # Ghi nhãn
        anns = imgid2anns.get(img_id, [])
        label_lines = []
        for ann in anns:
            cat_name = categories[ann['category_id']]
            if cat_name not in class2idx:
                print(f"Class {cat_name} không có trong danh sách class, bỏ qua annotation này.")
                continue
            class_id = class2idx[cat_name]
            x_c, y_c, w, h = coco2yolo_bbox(ann['bbox'], img['width'], img['height'])
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        # Ghi file label
        label_path = os.path.join(lbl_dst, os.path.splitext(file_name)[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

if not found_any:
    print('Không tìm thấy bất kỳ file annotation nào phù hợp. Hãy kiểm tra lại đường dẫn và tên file!')
else:
    # Tạo file log
    with open(LOG_FILE, 'w') as f:
        f.write('Merged increase_dataset successfully!\n')
    print("Đã merge xong increase_dataset vào dataset chính. Bạn có thể xóa thư mục increase_dataset.") 