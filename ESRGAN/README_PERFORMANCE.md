# Video Enhancement Performance Optimization

This directory contains optimized scripts for achieving higher FPS (>30) in real-time video enhancement using the student model.

## 🚀 Quick Start

### 1. Test Performance (No Webcam Required)
First, test your system's performance with different configurations:

```bash
# Compare all performance configurations
python test_performance.py --compare-all

# Test specific configuration
python test_performance.py --target ultra_fast

# List all available targets
python test_performance.py --list-targets
```

### 2. Run Video Enhancement
Choose the appropriate script based on your needs:

#### Option A: Configurable Script (Recommended)
```bash
# Run with fast configuration (45 FPS target)
python video_test_configurable.py --target fast

# Run with ultra-fast configuration (60 FPS target)
python video_test_configurable.py --target ultra_fast

# List all available targets
python video_test_configurable.py --list-targets
```

#### Option B: Ultra-Fast Script
```bash
# Maximum speed optimization
python video_test_ultra_fast.py
```

#### Option C: Optimized Original Script
```bash
# Improved version of original script
python video_test.py
```

## 📊 Performance Targets

| Configuration | Target FPS | Input Resolution | Quality | Use Case |
|---------------|------------|------------------|---------|----------|
| `ultra_fast`  | 60 FPS     | 128x128          | Lower   | Maximum speed |
| `fast`        | 45 FPS     | 256x256          | Medium  | Balanced performance |
| `standard`    | 30 FPS     | 384x384          | Good    | Standard quality |
| `quality`     | 15 FPS     | 512x512          | High    | Maximum quality |

## 🔧 Key Optimizations

### 1. **Resolution Reduction**
- Reduced input resolution from 512x512 to 128x128 for ultra-fast mode
- Faster processing with acceptable quality trade-off

### 2. **CUDA Optimizations**
- Enabled `cudnn.benchmark` for faster convolutions
- Disabled deterministic mode for speed
- High precision matrix multiplication

### 3. **Memory Optimizations**
- Non-blocking tensor transfers
- Reduced buffer sizes
- Efficient memory management

### 4. **Processing Optimizations**
- Optional sharpening skip for maximum speed
- Optimized preprocessing pipeline
- Reduced post-processing overhead

### 5. **Camera Optimizations**
- MJPG format for faster capture
- Minimal buffer sizes
- High FPS camera settings

## 🎯 Achieving >30 FPS

### For Systems with Good GPU (8GB+ VRAM):
```bash
python video_test_configurable.py --target ultra_fast
```
Expected: 50-70 FPS

### For Systems with Moderate GPU (6-8GB VRAM):
```bash
python video_test_configurable.py --target fast
```
Expected: 35-50 FPS

### For Systems with Limited GPU (<6GB VRAM):
```bash
python video_test_configurable.py --target standard
```
Expected: 25-40 FPS

## 📈 Performance Monitoring

All scripts display real-time performance metrics:
- Current FPS
- Average FPS
- Inference time (ms)
- Target FPS status

## 🔍 Troubleshooting

### Low FPS Issues:
1. **Check GPU memory**: Ensure sufficient VRAM
2. **Reduce resolution**: Use `ultra_fast` or `fast` targets
3. **Skip sharpening**: Enable in configuration
4. **Close other applications**: Free up GPU resources

### Camera Issues:
1. **Check camera permissions**: Ensure webcam access
2. **Try different camera index**: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
3. **Update camera drivers**: Ensure latest drivers

### Memory Issues:
1. **Reduce batch size**: Already optimized in scripts
2. **Lower resolution**: Use smaller input sizes
3. **Close other GPU applications**: Free up VRAM

## 📝 Configuration Customization

Edit `performance_config.py` to customize settings:

```python
# Example: Create custom configuration
RESOLUTION_CONFIGS['custom'] = {
    'input_size': (192, 192),  # Custom resolution
    'output_size': (640, 480),
    'skip_sharpening': False,
    'use_mixed_precision': True,
    'buffer_size': 2
}
```

## 🏆 Performance Tips

1. **Use CUDA**: Ensure PyTorch with CUDA support
2. **Monitor GPU usage**: Use `nvidia-smi` to check utilization
3. **Optimize system**: Close unnecessary applications
4. **Update drivers**: Keep GPU drivers current
5. **Test different targets**: Find optimal balance for your system

## 📊 Expected Results

Based on typical hardware configurations:

| GPU Type | VRAM | Expected FPS (ultra_fast) | Expected FPS (fast) |
|----------|------|---------------------------|---------------------|
| RTX 4090 | 24GB | 80-100 FPS               | 60-80 FPS          |
| RTX 3080 | 10GB | 60-80 FPS                | 45-60 FPS          |
| RTX 3070 | 8GB  | 50-70 FPS                | 35-50 FPS          |
| GTX 1660 | 6GB  | 30-50 FPS                | 25-40 FPS          |

## 🎮 Usage Examples

### Real-time Gaming Enhancement:
```bash
python video_test_configurable.py --target ultra_fast
```

### Quality-focused Enhancement:
```bash
python video_test_configurable.py --target quality
```

### Balanced Performance:
```bash
python video_test_configurable.py --target fast
```

## 📞 Support

If you're still not achieving >30 FPS:
1. Run performance test first: `python test_performance.py --compare-all`
2. Check your GPU specifications
3. Try different configuration targets
4. Consider hardware upgrades if needed

The scripts are designed to automatically select the best settings for your hardware while maintaining the >30 FPS target. 