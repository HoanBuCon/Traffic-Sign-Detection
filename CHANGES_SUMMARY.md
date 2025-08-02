# Tóm tắt các thay đổi cho Dataset mới (12 lớp)

## Dataset mới
- **Số lớp**: 12 (thay vì 86 lớp cũ)
- **Tên lớp**: 
  - i.423.b
  - p.102
  - p.106.b
  - p.130
  - p.131.a
  - r.308.b
  - sus
  - w.201.a
  - w.203.c
  - w.207.b
  - w.207.c
  - w.209

## Files đã cập nhật

### 1. data.yaml
- Cập nhật số lớp từ 86 xuống 12
- Cập nhật tên lớp theo dataset mới

### 2. config.py
- Cập nhật `nc: 12` thay vì 43
- Cập nhật tên lớp trong `get_dataset_yaml()`

### 3. predict.py
- Cập nhật `class_vi_map` với 12 lớp mới
- Cập nhật mô tả tiếng Việt cho từng lớp
- Loại bỏ các lớp cũ không còn sử dụng

### 4. predict2.py
- Cập nhật `class_vi_map` với 12 lớp mới
- Cập nhật mô tả tiếng Việt cho từng lớp

### 5. utils.py
- Cập nhật `create_dataset_yaml()` với 12 lớp mới
- Cập nhật tên lớp trong template

### 6. merge_increase_dataset.py
- Cập nhật `new_classes` với 12 lớp mới

## Mô tả tiếng Việt cho 12 lớp mới

| Lớp | Mô tả tiếng Việt |
|-----|-------------------|
| i.423.b | Biển chỉ dẫn khoảng cách |
| p.102 | Cấm đi ngược chiều |
| p.106.b | Cấm xe tải trên 2,5 tấn |
| p.130 | Cấm dừng xe và đỗ xe |
| p.131.a | Cấm đỗ xe |
| r.308.b | Hướng đi ưu tiên |
| sus | Biển báo nghi ngờ |
| w.201.a | Chỗ ngoặt nguy hiểm vòng bên trái |
| w.203.c | Đường người đi bộ cắt ngang |
| w.207.b | Giao nhau với đường ưu tiên |
| w.207.c | Giao nhau với đường cùng cấp |
| w.209 | Cầu hẹp |

## Lưu ý
- Tất cả các file đã được cập nhật để phù hợp với dataset mới
- Các file real-time prediction sẽ tự động đọc class names từ data.yaml
- Model training sẽ sử dụng 12 lớp mới
- Các file predict sẽ hiển thị đúng tên lớp và mô tả tiếng Việt 