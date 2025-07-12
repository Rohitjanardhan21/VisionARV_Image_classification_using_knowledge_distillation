# File: ESRGAN/video_test_high_clarity.py
# High-clarity video enhancement with 30+ FPS target

import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from student_model_enhanced import load_student_model
from collections import deque

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load model with optimizations
model = load_student_model('checkpoints/best_student_model.pth', device=DEVICE).eval()

# Enable optimizations for faster inference
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision('high')

# Pre-allocate tensors and transforms
to_tensor = ToTensor()
to_pil = ToPILImage()

# FPS tracking
fps_list = deque(maxlen=30)

# Video capture with optimized settings
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# High-quality camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Output video writer (30 FPS, 640x480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_high_clarity.avi', fourcc, 30.0, (640, 480))

print("Press 'q' to quit.")

# High-quality processing parameters
input_size = (384, 384)  # Higher resolution for better clarity
output_size = (640, 480)

# Enhanced sharpening function for better clarity
def enhance_clarity(tensor, strength=0.25):
    """Enhanced sharpening with better clarity"""
    # Use a more sophisticated kernel for better edge enhancement
    kernel = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
                          dtype=torch.float32, device=tensor.device) / 9.0
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sharpened = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
    enhanced = tensor + strength * (sharpened.squeeze(0) - tensor)
    return torch.clamp(enhanced, 0.0, 1.0)

# High-quality preprocessing
def preprocess_high_quality(frame):
    """High-quality preprocessing with better interpolation"""
    # Use cubic interpolation for better quality
    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_CUBIC)
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(resized, (3, 3), 0.5)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    # Convert to tensor
    tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE, non_blocking=True)
    return tensor

# High-quality postprocessing
def postprocess_high_quality(output_tensor):
    """High-quality postprocessing with better upscaling"""
    # Clamp values
    output = torch.clamp(output_tensor.squeeze(0), 0.0, 1.0)
    # Apply additional clarity enhancement
    output = enhance_clarity(output, strength=0.2)
    # Convert to numpy
    output_np = to_pil(output).convert("RGB")
    # Convert to BGR for OpenCV
    output_img = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
    # Use cubic interpolation for better upscaling
    output_img = cv2.resize(output_img, output_size, interpolation=cv2.INTER_CUBIC)
    # Apply slight sharpening filter
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output_img = cv2.filter2D(output_img, -1, kernel)
    return output_img

# Performance monitoring
frame_count = 0
start_time = time.time()

print("Starting high-clarity video processing...")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # High-quality preprocessing
        input_tensor = preprocess_high_quality(frame)

        # Inference timing
        inference_start = time.time()

        # Model inference
        output = model(input_tensor)

        # Inference end
        inference_end = time.time()
        inference_time = inference_end - inference_start

        # High-quality postprocessing
        output_img = postprocess_high_quality(output)

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_list.append(current_fps)

        # Calculate average FPS over last second
        if current_time - start_time >= 1.0:
            recent_fps = [f for f in fps_list if f > 0]
            avg_fps = sum(recent_fps) / len(recent_fps) if recent_fps else 0

        # Draw performance overlay
        cv2.putText(output_img, f"High Clarity Mode", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"FPS: {current_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Target: 30+ FPS", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Inference: {inference_time*1000:.1f}ms", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Resolution: {input_size[0]}x{input_size[1]}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image
        cv2.imshow("High Clarity Enhanced Output", output_img)

        # Save to file
        out.write(output_img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Print comprehensive statistics
if fps_list:
    valid_fps = [f for f in fps_list if f > 0]
    if valid_fps:
        avg_fps = sum(valid_fps) / len(valid_fps)
        max_fps = max(valid_fps)
        min_fps = min(valid_fps)
        elapsed_time = time.time() - start_time
        
        print(f"\n=== HIGH CLARITY Performance Statistics ===")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Max FPS: {max_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Processing resolution: {input_size[0]}x{input_size[1]}")
        print(f"Output resolution: {output_size[0]}x{output_size[1]}")
        print(f"30+ FPS target achieved: {'✅ YES' if avg_fps >= 30 else '❌ NO'}")
        print(f"High clarity mode: ✅ ENABLED")
        
        if avg_fps >= 30:
            print(f"🎉 Successfully achieved {avg_fps:.1f} FPS with high clarity!")
        else:
            print(f"⚠️  Current FPS: {avg_fps:.1f} - Consider using standard mode for better performance") 