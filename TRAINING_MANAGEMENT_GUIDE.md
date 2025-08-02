# Hướng dẫn sử dụng Hệ thống Quản lý Training

## 🚀 Tổng quan

Hệ thống quản lý training mới được thiết kế để tổ chức và theo dõi tất cả các lần training một cách có hệ thống. Mỗi lần training sẽ được lưu trong một thư mục riêng với cấu trúc rõ ràng.

## 📁 Cấu trúc thư mục

```
training_history/
├── training_log.json          # File log chính
├── performance_chart.png      # Biểu đồ hiệu suất
├── training_history.csv       # Xuất dữ liệu CSV
├── train1/                   # Session training 1
│   ├── weights/              # Trọng số model
│   │   ├── best.pt
│   │   └── last.pt
│   ├── plots/                # Biểu đồ training
│   │   ├── confusion_matrix.png
│   │   ├── labels.jpg
│   │   └── results.png
│   ├── logs/                 # Log files
│   ├── configs/              # Cấu hình training
│   │   └── training_config.json
│   ├── results/              # Kết quả validation
│   │   └── training_results.csv
│   ├── samples/              # Sample predictions
│   ├── augmented_data/       # Dữ liệu đã augment
│   └── checkpoints/          # Checkpoints
├── train2/                   # Session training 2
│   └── ...
└── train3_continue_from_1/   # Session tiếp tục từ train1
    └── ...
```

## 🛠️ Cách sử dụng

### 1. **Bắt đầu training mới**

```bash
python train.py
```

Hệ thống sẽ tự động:
- Tạo thư mục `train1/` (hoặc số tiếp theo)
- Tạo cấu trúc thư mục con
- Lưu cấu hình training
- Bắt đầu training với augmentation

### 2. **Quản lý training sessions**

```bash
python manage_training.py
```

Các lệnh có sẵn:
- `list` - Liệt kê tất cả sessions
- `list -d` - Liệt kê với chi tiết
- `best` - Hiển thị session tốt nhất
- `summary` - Tạo báo cáo tổng hợp
- `export` - Xuất ra CSV
- `copy <session_num>` - Copy session
- `delete <session_num>` - Xóa session

### 3. **Tiếp tục training từ session trước**

Hệ thống tự động phát hiện và tiếp tục từ session cuối cùng:

```bash
python train.py
```

## 📊 Thông tin được lưu trữ

### **Mỗi session training bao gồm:**

#### **1. Trọng số Model (`weights/`)**
- `best.pt` - Model tốt nhất
- `last.pt` - Model cuối cùng

#### **2. Biểu đồ Training (`plots/`)**
- Confusion matrix
- Loss curves
- mAP curves
- Sample predictions

#### **3. Cấu hình (`configs/`)**
- Thông tin session
- Cấu hình model
- Cấu hình augmentation
- Thông tin dataset

#### **4. Kết quả (`results/`)**
- Training metrics
- Validation results
- Performance statistics

#### **5. Logs (`logs/`)**
- Training logs
- Error logs
- Debug information

## 🔍 Theo dõi hiệu suất

### **1. Xem lịch sử training**

```bash
python manage_training.py
# Sau đó gõ: list
```

### **2. Tìm session tốt nhất**

```bash
python manage_training.py
# Sau đó gõ: best
```

### **3. Tạo báo cáo tổng hợp**

```bash
python manage_training.py
# Sau đó gõ: summary
```

### **4. Xuất dữ liệu**

```bash
python manage_training.py
# Sau đó gõ: export
```

## 📈 Biểu đồ và Báo cáo

### **Performance Chart**
- Biểu đồ mAP50 và mAP50-95 qua các sessions
- Tự động tạo khi có ít nhất 2 sessions hoàn thành

### **Summary Report**
- Tổng số sessions
- Sessions thành công/thất bại
- Hiệu suất tốt nhất
- Hiệu suất trung bình

### **CSV Export**
- Dữ liệu chi tiết tất cả sessions
- Có thể import vào Excel hoặc phân tích khác

## 🔧 Tùy chỉnh

### **1. Thay đổi cấu trúc thư mục**

Trong `train.py`, chỉnh sửa `create_training_directory_structure()`:

```python
subdirs = [
    'weights',           # Lưu trọng số model
    'plots',            # Lưu biểu đồ training
    'logs',             # Lưu log files
    'configs',          # Lưu cấu hình training
    'results',          # Lưu kết quả validation
    'samples',          # Lưu sample predictions
    'augmented_data',   # Lưu dữ liệu đã augment
    'checkpoints'       # Lưu checkpoints
]
```

### **2. Thêm thông tin session**

Trong `train.py`, chỉnh sửa `create_training_config_file()`:

```python
config_data = {
    'session_info': session_info,
    'model_config': {...},
    'augmentation_config': {...},
    'dataset_config': {...},
    # Thêm thông tin mới ở đây
}
```

## ⚠️ Lưu ý quan trọng

### **1. Backup dữ liệu**
- Luôn backup thư mục `training_history/` trước khi xóa
- Sử dụng `copy` command để tạo bản sao session quan trọng

### **2. Quản lý dung lượng**
- Mỗi session có thể chiếm vài GB
- Xóa sessions cũ nếu cần tiết kiệm dung lượng
- Sử dụng `delete` command cẩn thận

### **3. Tiếp tục training**
- Hệ thống tự động tìm session cuối cùng
- Có thể chỉ định session cụ thể để tiếp tục
- Kiểm tra trọng số trước khi tiếp tục

## 🎯 Best Practices

### **1. Đặt tên session rõ ràng**
- Sử dụng mô tả trong config file
- Ghi chú về thay đổi cấu hình

### **2. Theo dõi thường xuyên**
- Kiểm tra performance chart
- So sánh các sessions
- Ghi chú về cải tiến

### **3. Backup định kỳ**
- Backup `training_log.json`
- Backup sessions quan trọng
- Lưu trữ external

## 🚀 Quick Start

1. **Bắt đầu training đầu tiên:**
```bash
python train.py
```

2. **Xem lịch sử:**
```bash
python manage_training.py
# Gõ: list
```

3. **Tiếp tục training:**
```bash
python train.py
```

4. **Tạo báo cáo:**
```bash
python manage_training.py
# Gõ: summary
```

Hệ thống này sẽ giúp bạn quản lý training một cách chuyên nghiệp và theo dõi tiến độ một cách hiệu quả! 🎯 