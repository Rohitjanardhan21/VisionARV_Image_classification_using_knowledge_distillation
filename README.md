since , the datset is too large to be uploaded , the link to the datset is provided here. https://drive.google.com/drive/u/0/folders/1XCOPxzSVdAg6GcX_ssdbTs7mu--zbIf3?lfhs=2
# Intel Motion Deblurring Project

Real-time video deblurring using knowledge distillation with teacher-student architecture. This project implements motion blur removal for live video streams using a lightweight student model trained through knowledge distillation.

## Required Libraries

Install the following libraries in order:

```bash
# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core libraries
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install Pillow>=8.3.0
pip install matplotlib>=3.4.0
pip install tqdm>=4.62.0
pip install tensorboard>=2.7.0
pip install scikit-image>=0.18.0
pip install lpips>=0.1.4
```

## Project Structure

```
Intel project/
├── data/Dataset/Split/ (train/val/test blurry/sharp pairs)
├── checkpoints/
│   ├── motion_deblurring_teacher_best.pth
│   └── best_student_model.pth
├── ESRGAN/
│   ├── training_teacher.py
│   ├── train_kd.py
│   ├── student_model_enhanced.py
│   ├── post_train_eval_teacher.py
│   ├── post_training_eval_student.py
│   ├── student_visualize_tkinter.py
│   ├── video_test_high_clarity.py
│   ├── video_test_configurable.py
│   ├── test_performance.py
│   └── performance_config.py
├── blur_dataset.py
├── split_dataset.py
└── preprocess.py
```

## Code Descriptions

### Data Preparation
- `split_dataset.py` - Splits dataset into train/validation/test sets
- `preprocess.py` - Preprocesses images for training
- `blur_dataset.py` - Dataset loading utilities

### Training Pipeline
- `training_teacher.py` - Trains the NAFNet teacher model on motion blur dataset
- `train_kd.py` - Trains lightweight student model using knowledge distillation from teacher
- `student_model_enhanced.py` - Student model architecture definition

### Post-Training Evaluation
- `post_train_eval_teacher.py` - Evaluates trained teacher model performance
- `post_training_eval_student.py` - Evaluates trained student model performance

### Visualization
- `student_visualize_tkinter.py` - GUI application for visualizing student model outputs

### Video Enhancement
- `video_test_high_clarity.py` - High-clarity real-time video deblurring (384x384 input)
- `video_test_configurable.py` - Configurable video enhancement with multiple performance modes
- `test_performance.py` - Performance testing without webcam requirement
- `performance_config.py` - Performance configuration settings

## Execution Order

### Step 1: Data Preparation
```bash
# Split dataset into train/val/test
python split_dataset.py

# Preprocess images if needed
python preprocess.py
```

### Step 2: Model Training
```bash
cd ESRGAN

# Train teacher model (NAFNet)
python training_teacher.py

# Train student model using knowledge distillation
python train_kd.py
```

### Step 3: Post-Training Evaluation
```bash
# Evaluate teacher model performance
python post_train_eval_teacher.py

# Evaluate student model performance
python post_training_eval_student.py
```

### Step 4: Model Visualization
```bash
# Launch GUI for visualizing student model outputs
python student_visualize_tkinter.py
```

### Step 5: Performance Testing
```bash
# Test system performance (no webcam required)
python test_performance.py --compare-all

# Test specific configuration
python test_performance.py --target fast
```

### Step 6: Real-time Video Enhancement
```bash
# High-clarity mode (recommended)
python video_test_high_clarity.py

# Configurable modes
python video_test_configurable.py --target ultra_fast  # 60 FPS
python video_test_configurable.py --target fast        # 45 FPS
python video_test_configurable.py --target high_clarity # 35 FPS
python video_test_configurable.py --target standard    # 30 FPS
python video_test_configurable.py --target quality     # 15 FPS
```

## Output Files

### Training Outputs
- `checkpoints/motion_deblurring_teacher_best.pth` - Trained teacher model
- `checkpoints/best_student_model.pth` - Trained student model

### Evaluation Outputs
- Teacher model evaluation results and metrics
- Student model evaluation results and metrics

### Video Enhancement Outputs
- `output_high_clarity.avi` - High-clarity mode output
- `output_sharpened_[mode].avi` - Configurable mode outputs

## Hardware Requirements

- NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- Python 3.8 or higher
- Sufficient RAM for video processing
