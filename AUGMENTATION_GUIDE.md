# Hướng dẫn sử dụng Data Augmentation

## 🚀 Tổng quan

Hệ thống Data Augmentation đã được cải thiện với nhiều kỹ thuật tiên tiến để tăng cường dữ liệu training, giúp model có khả năng generalization tốt hơn.

## 📋 Các kỹ thuật Augmentation

### 1. **Geometric Transformations**
- **Rotation**: Xoay ảnh đến 30 độ
- **Horizontal Flip**: Lật ngang ảnh (50% probability)
- **Vertical Flip**: Lật dọc ảnh (10% probability)
- **Shift Scale Rotate**: Dịch chuyển, scale và xoay kết hợp
- **Perspective**: Biến đổi phối cảnh

### 2. **Color & Brightness Augmentation**
- **Random Brightness/Contrast**: Thay đổi độ sáng và độ tương phản
- **Random Gamma**: Điều chỉnh gamma correction
- **Hue Saturation Value**: Thay đổi màu sắc, độ bão hòa, độ sáng
- **CLAHE**: Cải thiện ảnh thiếu sáng

### 3. **Noise & Blur Augmentation**
- **Gaussian Noise**: Thêm nhiễu Gaussian
- **ISO Noise**: Mô phỏng nhiễu camera
- **Multiplicative Noise**: Nhiễu nhân
- **Motion Blur**: Làm mờ chuyển động
- **Median Blur**: Làm mờ trung vị
- **Gaussian Blur**: Làm mờ Gaussian

### 4. **Weather & Lighting Effects**
- **Random Rain**: Thêm hiệu ứng mưa
- **Random Fog**: Thêm hiệu ứng sương mù
- **Random Sun Flare**: Thêm hiệu ứng flare mặt trời

### 5. **Occlusion & Cutout**
- **Coarse Dropout**: Cắt bỏ các vùng ngẫu nhiên
- **Grid Dropout**: Cắt bỏ theo lưới

### 6. **Elastic & Optical Distortions**
- **Elastic Transform**: Biến dạng đàn hồi
- **Optical Distortion**: Biến dạng quang học

## 🛠️ Cách sử dụng

### 1. **Tự động tăng cường dataset**

```bash
python auto_augment_dataset.py
```

Script này sẽ:
- Tạo 5 augmented samples cho mỗi ảnh training
- Tạo 2 augmented samples cho mỗi ảnh validation
- Tạo 1 augmented sample cho mỗi ảnh test
- Lưu vào thư mục `augmented_dataset/`

### 2. **Demo augmentation**

```bash
python demo_augmentation.py
```

Script này sẽ:
- Hiển thị các kỹ thuật augmentation khác nhau
- Tạo grid ảnh so sánh
- Lưu kết quả vào thư mục `augmentation_demo/`

### 3. **Training với augmentation**

```bash
python train.py
```

Model sẽ tự động sử dụng các kỹ thuật augmentation đã cấu hình trong `config.py`.

## ⚙️ Cấu hình Augmentation

Các tham số augmentation có thể điều chỉnh trong `config.py`:

```python
# Augmentation Strength
AUGMENTATION_STRENGTH = 0.8  # Độ mạnh tổng thể (0.0-1.0)

# Geometric
ROTATION = 30  # Góc xoay tối đa
HORIZONTAL_FLIP = 0.5  # Xác suất lật ngang
VERTICAL_FLIP = 0.1  # Xác suất lật dọc

# Color & Brightness
BRIGHTNESS_LIMIT = 0.3  # Giới hạn thay đổi độ sáng
CONTRAST_LIMIT = 0.3  # Giới hạn thay đổi độ tương phản
HUE_LIMIT = 20  # Giới hạn thay đổi màu sắc

# Weather Effects
RAIN_PROBABILITY = 0.2  # Xác suất thêm mưa
FOG_PROBABILITY = 0.2  # Xác suất thêm sương mù
SUNFLARE_PROBABILITY = 0.2  # Xác suất thêm flare
```

## 📊 Kết quả mong đợi

### Trước augmentation:
- Dataset gốc: ~1000 ảnh
- Khả năng generalization thấp
- Overfitting dễ xảy ra

### Sau augmentation:
- Dataset mở rộng: ~5000-6000 ảnh
- Đa dạng về điều kiện môi trường
- Model robust hơn với các điều kiện thực tế

## 🔧 Tùy chỉnh nâng cao

### 1. **Thêm kỹ thuật augmentation mới**

Trong `utils.py`, thêm vào `DataAugmentation` class:

```python
# Thêm vào augmentation_pipeline
A.YourCustomAugmentation(parameters, p=0.5),
```

### 2. **Điều chỉnh probability**

Thay đổi xác suất trong `config.py`:

```python
CUSTOM_AUGMENTATION_PROBABILITY = 0.3
```

### 3. **Tạo augmentation pipeline riêng**

```python
custom_pipeline = A.Compose([
    A.Rotate(limit=45, p=0.8),
    A.RandomBrightnessContrast(p=0.9),
    # Thêm các kỹ thuật khác
])
```

## ⚠️ Lưu ý quan trọng

1. **Không augment quá mạnh**: Có thể làm mất thông tin quan trọng
2. **Kiểm tra kết quả**: Luôn xem xét các ảnh đã augment
3. **Validation**: Sử dụng ít augmentation cho validation
4. **Performance**: Augmentation có thể làm chậm training

## 📈 Monitoring

Theo dõi quá trình augmentation:

```python
# Trong train.py
print(f"Augmentation: {self.config.AUGMENTATION}")
print(f"Augmentation Strength: {self.config.AUGMENTATION_STRENGTH}")
```

## 🎯 Best Practices

1. **Bắt đầu với augmentation nhẹ**: Tăng dần độ mạnh
2. **Test trên subset**: Thử nghiệm trước khi áp dụng toàn bộ
3. **Monitor metrics**: Theo dõi mAP, loss để điều chỉnh
4. **Balance classes**: Đảm bảo augmentation cân bằng giữa các lớp

## 🚀 Quick Start

1. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

2. **Chạy auto augmentation**:
```bash
python auto_augment_dataset.py
```

3. **Train model**:
```bash
python train.py
```

4. **Demo kết quả**:
```bash
python demo_augmentation.py
```

Hệ thống augmentation này sẽ giúp model của bạn robust hơn và có khả năng generalization tốt hơn trong các điều kiện thực tế! 🎯 