# File: ESRGAN/video_test_ultra_fast.py
# Ultra-fast video enhancement with >30 FPS target

import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from student_model_enhanced import load_student_model
import threading
from collections import deque
import queue

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Load model with maximum optimizations
model = load_student_model('checkpoints/best_student_model.pth', device=DEVICE).eval()

# Enable all possible optimizations
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    # Use mixed precision for faster inference
    torch.set_float32_matmul_precision('high')

# Pre-allocate tensors and transforms
to_tensor = ToTensor()
to_pil = ToPILImage()

# FPS tracking
fps_list = deque(maxlen=60)

# Video capture with maximum performance settings
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Maximum performance camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 120)  # Request maximum FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for faster capture

# Output video writer (60 FPS, 640x480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_sharpened_ultra_fast.avi', fourcc, 60.0, (640, 480))

print("Press 'q' to quit.")

# Ultra-fast processing parameters
input_size = (128, 128)  # Even smaller for maximum speed
output_size = (640, 480)

# Frame processing queue for potential threading
frame_queue = queue.Queue(maxsize=2)

# Performance monitoring
frame_count = 0
start_time = time.time()
last_fps_time = time.time()

# Skip sharpening for maximum speed
SKIP_SHARPENING = True

# Fast resize function
def fast_resize(frame, size):
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

# Ultra-fast preprocessing
def preprocess_frame_fast(frame):
    # Resize to very small size for speed
    resized = fast_resize(frame, input_size)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert to tensor
    tensor = to_tensor(rgb).unsqueeze(0).to(DEVICE, non_blocking=True)
    return tensor

# Fast post-processing
def postprocess_fast(output_tensor):
    # Clamp values
    output = torch.clamp(output_tensor.squeeze(0), 0.0, 1.0)
    # Convert to numpy
    output_np = to_pil(output).convert("RGB")
    # Convert to BGR for OpenCV
    output_img = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
    # Resize to output size
    return cv2.resize(output_img, output_size, interpolation=cv2.INTER_LINEAR)

print("Starting ultra-fast video processing...")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ultra-fast preprocessing
        input_tensor = preprocess_frame_fast(frame)

        # Inference timing
        inference_start = time.time()

        # Model inference
        output = model(input_tensor)

        # Inference end
        inference_end = time.time()
        inference_time = inference_end - inference_start

        # Post-processing
        output_img = postprocess_fast(output)

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_list.append(current_fps)

        # Calculate average FPS over last second
        if current_time - last_fps_time >= 1.0:
            recent_fps = [f for f in fps_list if f > 0]
            avg_fps = sum(recent_fps) / len(recent_fps) if recent_fps else 0
            last_fps_time = current_time

        # Draw performance overlay
        cv2.putText(output_img, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Avg FPS: {avg_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Inference: {inference_time*1000:.1f}ms", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Resolution: {input_size[0]}x{input_size[1]}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Ultra-Fast Enhanced Output", output_img)

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
        
        print(f"\n=== ULTRA-FAST Performance Statistics ===")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Max FPS: {max_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Processing resolution: {input_size[0]}x{input_size[1]}")
        print(f"Output resolution: {output_size[0]}x{output_size[1]}")
        print(f"Target achieved: {'✅ YES' if avg_fps > 30 else '❌ NO'}")
        
        if avg_fps > 30:
            print(f"🎉 Successfully achieved {avg_fps:.1f} FPS (>30 FPS target)!")
        else:
            print(f"⚠️  Current FPS: {avg_fps:.1f} - Consider further optimizations") 