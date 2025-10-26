# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Common commands

- Setup environment
  - Create venv (optional)
    ```bash path=null start=null
    python -m venv .venv
    # Windows (PowerShell)
    .\.venv\Scripts\Activate.ps1
    # macOS/Linux
    source .venv/bin/activate
    ```
  - Install dependencies
    ```bash path=null start=null
    pip install -r requirements.txt
    ```

- Dataset prep (expected layout)
  - Place your dataset under `dataset/` using YOLO format:
    - Images: `dataset/images/train`, `dataset/images/val`, `dataset/images/test` (test optional)
    - Labels: `dataset/labels/train`, `dataset/labels/val`, `dataset/labels/test`
  - If you have flat `dataset/images` and `dataset/labels`, you can split into train/val with:
    ```bash path=null start=null
    python src/split_dataset.py
    ```

- Required data.yaml (repo root)
  - Training/inference expect a `data.yaml` at repo root. If missing, create one like:
    ```yaml path=null start=null
    # Dataset configuration
    path: ./dataset
    train: images/train
    val: images/val
    test: images/test  # optional
    nc: 12             # number of classes (adjust to your dataset)
    names:             # class codes or names by index
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
    descriptions:      # human-readable Vietnamese labels aligned by index
      - Đường kẻ dành cho người đi bộ
      - Cấm đi ngược chiều
      - Cấm xe tải trên N tấn
      - Cấm dừng và đỗ xe
      - Cấm đỗ xe
      - Tuyến đường cầu vượt cắt qua
      - TEXT SIGN
      - Cảnh báo khúc cua nguy hiểm bên trái
      - Cảnh báo đường hẹp
      - Cảnh báo giao nhau với đường không ưu tiên bên phải
      - Cảnh báo giao nhau với đường không ưu tiên bên trái
      - Cảnh báo đến khu vực đèn tín hiệu giao thông
    ```

- Train (saves best/last and organizes weights)
  ```bash path=null start=null
  # From repo root
  python src/training/train.py
  ```
  - Outputs: `runs/traffic_sign_detection/...`; after training, `best.pt` and `last.pt` are moved to `all_weight/trainX/` (X auto-increments).

- Validate/Test separately (optional)
  ```bash path=null start=null
  python -c "from src.training.train import TrafficSignTrainer; t=TrafficSignTrainer(); t.setup_training(); t.validate(); t.test()"
  ```

- Batch image inference (reads from `input/`, writes to `output/predictX/`)
  ```bash path=null start=null
  # Ensure latest trained weight exists under all_weight/trainX/best.pt
  # Place images in ./input
  python src/pipeline/predict.py
  ```

- Real-time webcam inference
  - Basic smoothing:
    ```bash path=null start=null
    python src/pipeline/real_time_predict.py
    ```
  - Advanced (SORT tracker + stronger smoothing):
    ```bash path=null start=null
    python src/pipeline/real_time_predict_smooth_advanced.py
    ```
  - Mobile-optimized variant:
    ```bash path=null start=null
    python src/pipeline/real_time_predict_smooth_advanced_mobile.py
    ```

- NLP-hybrid real-time (optional Hugging Face login)
  ```bash path=null start=null
  # Optional: create .env with HF_TOKEN=<your_token>
  python src/pipeline/real_time_predict_nlp_hybrid.py
  ```

- Linting/formatting: not configured in this repo.
- Tests: no test suite found in this repo.

## Architecture overview

High-level flow (YOLOv8-based):

- Training (`src/training`)
  - `train.py` defines `TrafficSignTrainer` that orchestrates:
    - `setup_training()`: ensures output dirs; expects a valid `data.yaml` at repo root.
    - `train()`: fine-tunes YOLOv8 model defined by `Config.MODEL_SIZE` with augmentation knobs and training hyperparameters from `Config`.
    - Post-train: moves `runs/traffic_sign_detection/weights/{best,last}.pt` into a versioned folder under `all_weight/trainX/`.
    - `validate()` and `test()`: evaluate on val/test splits using the same `data.yaml`.
  - `config.py` centralizes hyperparameters, dataset paths, and inference thresholds; also controls image enhancement toggles.
  - `core/utils.py::DataAugmentation` defines Albumentations pipeline used during training.

- Offline inference (`src/pipeline/predict.py`)
  - Auto-loads the latest `all_weight/trainX/best.pt` if no `model_path` is provided.
  - Reads class `names` and `descriptions` from root-level `data.yaml`.
  - Applies `ImageEnhancer` preprocessing, runs YOLOv8 inference, and renders results via `VisualizationUtils`.
  - Saves annotated images and structured JSON metadata to `output/predictX/{images,json}`.

- Real-time inference (`src/pipeline`)
  - `real_time_predict.py`: Basic frame-by-frame smoothing with label buffers.
  - `real_time_predict_smooth_advanced.py` and `real_time_predict_smooth_advanced_mobile.py`:
    - Integrate SORT tracker (`src/sort.py`) to maintain object IDs across frames.
    - Associate detections to tracks (IoU), smooth labels per tracked object with confidence-aware buffers, and overlay richer labels (class code + Vietnamese description + confidence).
    - Write video outputs to `real_time_output/`.
  - `real_time_predict_nlp_hybrid.py` (optional): augments CV with lightweight NLP/vision components from Hugging Face; supports `.env`-based `HF_TOKEN` and auto-login via `huggingface_hub` if present.

- Core utilities (`src/core/utils.py`)
  - `ImageEnhancer`: denoising, sharpening, gamma/CLAHE, low-light handling; used by inference pipelines.
  - `VisualizationUtils`: color-coded boxes and label overlays; can emit per-image detection JSON.
  - `FileUtils`: helpers for image discovery; includes a dataset YAML generator (standalone) that you can adapt if needed.

- Dataset helper
  - `src/split_dataset.py`: splits a flat `dataset/images` + `dataset/labels` into YOLO `train/val` subfolders in-place under `dataset/`.

## Notes and pitfalls

- Ensure a valid `data.yaml` exists at repo root before training/inferring; code reads `names` and `descriptions` from it, and YOLO requires `path/train/val` entries.
- Inference scripts auto-pick the latest weight from `all_weight/trainX/best.pt`; train at least once before running them.
- Place inputs under `input/`; outputs will be written under `output/` and videos under `real_time_output/`.
