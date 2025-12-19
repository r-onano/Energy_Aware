# Energy-Aware Perception Scheduling for Autonomous Vehicles

**A machine learning framework that reduces autonomous vehicle energy consumption by 26.4% while maintaining safety.**
---

##  Overview

This project implements an intelligent perception scheduling system that dynamically adjusts object detection model complexity based on real-time scene analysis. Instead of running the most powerful (and energy-intensive) detection models constantly, our system:

- **Analyzes scene complexity** using 12 visual features
- **Selects the right model** (YOLOv5 small/medium/large) for each frame
- **Maintains safety** through explicit override rules for critical scenarios
- **Saves energy** by 26.4% compared to always-heavy processing

### Key Results
-  **26.4% energy savings** vs constant heavy processing
-  **85.3% critical object detection** (pedestrians & cyclists)
-  **65.3% F1 score** with minimal accuracy loss
-  **360 MWh daily savings** at 1 million vehicle scale

---

## System Architecture

```
Input Frame → Feature Extraction → Complexity Prediction → Safety Override → Model Selection
                (12 features)      (Random Forest)         (3 rules)       (YOLOv5 s/m/l)
                    ↓                     ↓                     ↓                 ↓
                 6.8 ms               0.9 ms                0.3 ms          15-47 ms
```

**Total Scheduling Overhead:** 8.0 ms (17% of lightest model inference time)

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/r-onano/Energy_Aware.git
cd Energy_Aware
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt --break-system-packages
```

### 3. Download Datasets

**KITTI Dataset:**
```bash
# Download from: http://www.cvlibs.net/datasets/kitti/eval_object.php
# Required files:
# - data_object_image_2.zip (12 GB)
# - data_object_label_2.zip (5 MB)

# Extract to: data/raw/kitti/
mkdir -p data/raw/kitti
unzip data_object_image_2.zip -d data/raw/kitti/
unzip data_object_label_2.zip -d data/raw/kitti/
```

**BDD100K Dataset (Optional):**
```bash
# Download from: https://bdd-data.berkeley.edu/
# Required: 100K Images + Labels

# Extract to: data/raw/bdd100k/
```

### 4. Run the Pipeline

**Full Pipeline:**
```bash
# Preprocess data, train model, and evaluate
python run_pipeline.py --steps preprocess train evaluate
```

**Individual Steps:**
```bash
# 1. Preprocess KITTI dataset
python src/preprocessing/preprocess_multi_dataset.py --dataset kitti

# 2. Train Random Forest model
python src/models/train.py --model random_forest

# 3. Evaluate energy-aware scheduling
python src/evaluation/evaluate.py --num_frames 100
```

---

Generated visualizations will be saved in `data/results/`.

---

## Project Structure

```
Energy_Aware/
├── config_multi_dataset.yaml          # Configuration file
├── requirements.txt                    # Python dependencies
│
├── src/
│   ├── preprocessing/
│   │   └── preprocess_multi_dataset.py    # Feature extraction
│   ├── data_processing/
│   │   ├── feature_extraction.py          # 12 complexity features
│   │   ├── kitti_loader.py                # KITTI dataset loader
│   │   └── bdd100k_loader.py              # BDD100K dataset loader
│   ├── models/
│   │   ├── train.py                       # ML model training
│   │   └── predictor.py                   # Complexity predictor
│   ├── scheduling/
│   │   └── scheduler.py                   # Energy-aware scheduler
│   └── evaluation/
│       └── evaluate.py                    # Performance evaluation
│
├── data/                                  # Created after setup
│   ├── raw/                               # Downloaded datasets
│   ├── processed/                         # Preprocessed features
│   └── results/                           # Evaluation outputs
│
└── models/                                # Trained models
    └── random_forest_complexity_predictor.pkl
```

---

## Configuration

Edit `config_multi_dataset.yaml` to adjust:

```yaml
# Complexity thresholds
thresholds:
  skip: 0.1      # Reuse previous frame detection
  light: 0.3     # YOLOv5s (2.5J)
  medium: 0.6    # YOLOv5m (4.2J)
  heavy: 1.0     # YOLOv5l (6.8J)

# Dataset paths
datasets:
  kitti:
    image_dir: data/raw/kitti/training/image_2
    label_dir: data/raw/kitti/training/label_2
```

---

## Troubleshooting

### **Issue 1: Import Errors**
```
ImportError: cannot import name 'compute_complexity_score'
```
**Solution:** This function is a method, not standalone. Use:
```python
from src.data_processing.feature_extraction import FeatureExtractor
extractor = FeatureExtractor(config)
score = extractor.compute_complexity_score(features)
```

### **Issue 2: BDD100K Annotations Not Loading**
```
All images classified as "skip" (100%)
```
**Solution:** BDD100K uses `frame['objects']` not `frame['labels']`. Ensure you're using the corrected loader:
```python
# In bdd100k_loader.py, line 111:
objects = frame.get('objects', frame.get('labels', []))  # ✓ CORRECT
```

### **Issue 3: NVIDIA Driver Issues on Ubuntu**
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```
**Solution:**
```bash
# Check driver status
nvidia-smi

# If failed, reinstall drivers
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-driver-535
sudo reboot
```

### **Issue 4: pip Install Fails (Permission Denied)**
```
ERROR: Could not install packages due to an OSError
```
**Solution:** Use `--break-system-packages` flag:
```bash
pip install -r requirements.txt --break-system-packages
```

### **Issue 5: Out of Memory (GPU)**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use smaller model:
```python
# In evaluate.py
batch_size = 1  # Reduce from default
torch.cuda.empty_cache()  # Clear cache between runs
```

### **Issue 6: KeyError in Processed Data**
```
KeyError: 'X is not a file in the archive'
```
**Solution:** Old processed data used different keys. Delete and reprocess:
```bash
rm data/processed/kitti_processed.npz
python src/preprocessing/preprocess_multi_dataset.py --dataset kitti
```

### **Issue 7: PyTorch Version Mismatch**
```
ImportError: cannot import name '_six' from 'torch._six'
```
**Solution:** PyTorch version conflicts. Reinstall:
```bash
pip uninstall torch torchvision
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### **Issue 8: Slow Preprocessing**
```
Processing taking hours for KITTI dataset
```
**Solution:** This is normal. Expected times:
- KITTI: 18-25 minutes (7,481 images)
- BDD100K: 1.5-2 hours (70,000 images)

Use progress bar to monitor: `tqdm` installed automatically.

---

## Hardware Requirements

### **Minimum:**
- CPU: 4 cores (Intel i5 or AMD Ryzen 5)
- RAM: 8 GB
- Storage: 50 GB free space
- GPU: Not required (CPU mode available)

### **Recommended:**
- CPU: 8+ cores (Intel i9 or AMD Ryzen 9)
- RAM: 16+ GB
- Storage: 100 GB SSD
- GPU: NVIDIA RTX 3050+ (4GB VRAM)
- CUDA: Version 11.8+

**Note:** Our experiments used Lenovo i9-13905H + RTX 4050 (6GB VRAM).

---

## Author

**Cepher Onano**
- GitHub: [@r-onano](https://github.com/r-onano)
---
