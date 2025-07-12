# Intel Motion Deblurring Project

A comprehensive motion deblurring system using knowledge distillation with real-time video enhancement capabilities. This project implements a teacher-student architecture where a large teacher model (NAFNet) transfers knowledge to a lightweight student model for efficient real-time video deblurring.

## 🎯 Project Overview

This project consists of two main components:
1. **Training Pipeline**: Knowledge distillation from teacher to student model
2. **Real-time Video Enhancement**: Live video deblurring using the trained student model

## 📁 Project Structure

```
Intel project/
├── data/
│   └── Dataset/
│       └── Split/
│           ├── train/ (blurry/sharp pairs)
│           ├── val/ (blurry/sharp pairs)
│           └── test/ (blurry/sharp pairs)
├── checkpoints/
│   ├── motion_deblurring_teacher_best.pth
│   └── best_student_model.pth
├── ESRGAN/
│   ├── nafnet_teacher.py
│   ├── training_teacher.py
│   ├── train_kd.py
│   ├── student_model_enhanced.py
│   ├── post_train_eval_teacher.py
│   ├── post_training_eval_student.py
│   ├── video_test.py
│   ├── video_test_ultra_fast.py
│   ├── video_test_high_clarity.py
│   ├── video_test_configurable.py
│   ├── test_performance.py
│   ├── performance_config.py
│   ├── student_visualize_tkinter.py
│   ├── test_student_app.py
│   ├── video_test.py
│   ├── utils/
│   │   └── dataset.py
│   └── vgg_perceptual_multi.py
├── blur_dataset.py
├── split_dataset.py
├── preprocess.py
└── README.md
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
conda create -n intel_deblur python=3.8
conda activate intel_deblur

# Install required libraries
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Split your dataset into train/val/test
python split_dataset.py

# Preprocess images if needed
python preprocess.py
```

### 3. Training Pipeline

```bash
# Step 1: Train the teacher model (NAFNet)
cd ESRGAN
python training_teacher.py

# Step 2: Train the student model using knowledge distillation
python train_kd.py
```

### 4. Real-time Video Enhancement

```bash
# Test performance first (no webcam required)
python test_performance.py --compare-all

# Run high-clarity video enhancement
python video_test_high_clarity.py

# Or use configurable script
python video_test_configurable.py --target high_clarity
```

## 📋 Required Libraries

Create a `requirements.txt` file with the following dependencies:

```txt
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
matplotlib>=3.4.0
tqdm>=4.62.0
tensorboard>=2.7.0
scikit-image>=0.18.0
lpips>=0.1.4
```

## 🔧 Detailed Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended
- **RAM**: 16GB+ recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Intel-project

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install opencv-python numpy Pillow matplotlib tqdm tensorboard scikit-image lpips
```

## 📊 Training Pipeline

### Step 1: Teacher Model Training

```bash
cd ESRGAN
python training_teacher.py
```

**Purpose**: Trains the NAFNet teacher model on motion blur dataset
**Output**: `checkpoints/motion_deblurring_teacher_best.pth`

### Step 2: Knowledge Distillation

```bash
python train_kd.py
```

**Purpose**: Trains lightweight student model using teacher knowledge
**Output**: `checkpoints/best_student_model.pth`

### Step 3: Model Evaluation

```bash
# Evaluate teacher model
python post_train_eval_teacher.py

# Evaluate student model
python post_training_eval_student.py
```

## 🎥 Real-time Video Enhancement

### Performance Testing (No Webcam Required)

```bash
# Test all performance configurations
python test_performance.py --compare-all

# Test specific configuration
python test_performance.py --target fast

# List available targets
python test_performance.py --list-targets
```

### Video Enhancement Scripts

#### 1. High Clarity Mode (Recommended)
```bash
python video_test_high_clarity.py
```
- **Target**: 30+ FPS with maximum clarity
- **Input Resolution**: 384x384
- **Output**: `output_high_clarity.avi`

#### 2. Configurable Mode
```bash
# Ultra-fast (60 FPS target)
python video_test_configurable.py --target ultra_fast

# Fast (45 FPS target)
python video_test_configurable.py --target fast

# High clarity (35 FPS target)
python video_test_configurable.py --target high_clarity

# Standard (30 FPS target)
python video_test_configurable.py --target standard

# Quality (15 FPS target)
python video_test_configurable.py --target quality
```

#### 3. Ultra-Fast Mode
```bash
python video_test_ultra_fast.py
```
- **Target**: Maximum speed (60 FPS)
- **Input Resolution**: 128x128
- **Output**: `output_sharpened_ultra_fast.avi`

#### 4. Optimized Original
```bash
python video_test.py
```
- **Target**: Balanced performance
- **Input Resolution**: 256x256
- **Output**: `output_sharpened.avi`

## 📈 Performance Configurations

| Configuration | Target FPS | Input Resolution | Quality | Use Case |
|---------------|------------|------------------|---------|----------|
| `ultra_fast`  | 60 FPS     | 128x128          | Lower   | Maximum speed |
| `fast`        | 45 FPS     | 256x256          | Medium  | Balanced performance |
| `high_clarity`| 35 FPS     | 384x384          | High    | High clarity + 30+ FPS |
| `standard`    | 30 FPS     | 384x384          | Good    | Standard quality |
| `quality`     | 15 FPS     | 512x512          | Maximum | Maximum quality |

## 🎮 Usage Examples

### For Real-time Gaming Enhancement:
```bash
python video_test_configurable.py --target ultra_fast
```

### For Video Calls with Enhanced Clarity:
```bash
python video_test_high_clarity.py
```

### For Quality-focused Applications:
```bash
python video_test_configurable.py --target quality
```

### For Balanced Performance:
```bash
python video_test_configurable.py --target fast
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce input resolution in performance_config.py
# Or use ultra_fast configuration
python video_test_configurable.py --target ultra_fast
```

#### 2. Low FPS
```bash
# Test performance first
python test_performance.py --compare-all

# Use appropriate configuration for your hardware
python video_test_configurable.py --target fast
```

#### 3. Camera Not Working
```bash
# Check camera permissions
# Try different camera index in video scripts
# Update camera drivers
```

#### 4. Model Loading Errors
```bash
# Ensure checkpoints exist
ls checkpoints/
# Re-run training if needed
python training_teacher.py
python train_kd.py
```

### Performance Optimization

1. **GPU Memory**: Ensure sufficient VRAM (6GB+ recommended)
2. **Close Applications**: Free up GPU resources
3. **Update Drivers**: Keep GPU drivers current
4. **Monitor Usage**: Use `nvidia-smi` to check utilization

## 📊 Expected Results

Based on typical hardware configurations:

| GPU Type | VRAM | Expected FPS (ultra_fast) | Expected FPS (high_clarity) |
|----------|------|---------------------------|------------------------------|
| RTX 4090 | 24GB | 80-100 FPS               | 50-70 FPS                   |
| RTX 3080 | 10GB | 60-80 FPS                | 40-60 FPS                   |
| RTX 3070 | 8GB  | 50-70 FPS                | 35-50 FPS                   |
| GTX 1660 | 6GB  | 30-50 FPS                | 25-40 FPS                   |

## 🏆 Key Features

- **Real-time Video Deblurring**: Live motion blur removal
- **Knowledge Distillation**: Efficient teacher-student architecture
- **Multiple Performance Modes**: From ultra-fast to maximum quality
- **Universal Video Format**: AVI output for wide compatibility
- **Performance Monitoring**: Real-time FPS and quality metrics
- **Hardware Optimization**: Automatic CUDA optimizations

## 📝 Configuration

Customize performance settings in `ESRGAN/performance_config.py`:

```python
# Example: Create custom configuration
RESOLUTION_CONFIGS['custom'] = {
    'input_size': (192, 192),
    'output_size': (640, 480),
    'skip_sharpening': False,
    'use_mixed_precision': True,
    'buffer_size': 2
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NAFNet architecture for teacher model
- Knowledge distillation techniques
- OpenCV for video processing
- PyTorch for deep learning framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review performance configurations
3. Test with different hardware settings
4. Open an issue with detailed information

---

**Note**: This project requires a CUDA-capable GPU for optimal performance. CPU-only mode is supported but significantly slower. 