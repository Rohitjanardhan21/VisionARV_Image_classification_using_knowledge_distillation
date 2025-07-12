# File: ESRGAN/video_test.py

import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from student_model_enhanced import load_student_model
import threading
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

# Pre-allocate tensors and transforms
to_tensor = ToTensor()
to_pil = ToPILImage()

# FPS tracking
fps_list = deque(maxlen=30)  # Keep last 30 FPS readings

# Video capture with optimized settings
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Optimize camera settings for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS from camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size

# Output video writer (60 FPS, 640x480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_sharpened.avi', fourcc, 60.0, (640, 480))

print("Press 'q' to quit.")

# Optimized unsharp masking function
def sharpen_image_fast(tensor, strength=0.15):
    # Use a smaller, optimized kernel
    kernel = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                          dtype=torch.float32, device=tensor.device) / 6.0
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
    sharpened = tensor + strength * (tensor - blurred.squeeze(0))
    return torch.clamp(sharpened, 0.0, 1.0)

# Pre-allocate tensors for better performance
input_size = (256, 256)  # Reduced from 512x512 for faster processing
output_size = (640, 480)

# Frame buffer for potential batch processing
frame_buffer = deque(maxlen=3)

# Performance monitoring
frame_count = 0
start_time = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optimized preprocessing
        # Resize to smaller size for faster processing
        input_frame = cv2.resize(frame, input_size)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # Prepare tensor (optimized)
        input_tensor = to_tensor(input_frame).unsqueeze(0).to(DEVICE, non_blocking=True)

        # Inference start
        inference_start = time.time()

        # Model inference
        output = model(input_tensor).squeeze(0)
        output = torch.clamp(output, 0.0, 1.0)
        
        # Optional: Skip sharpening for even higher FPS
        # output = sharpen_image_fast(output, strength=0.15)
        
        output = output.cpu()

        # Inference end and FPS calculation
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        # Calculate FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        
        fps_list.append(current_fps)

        # Convert to displayable image (optimized)
        output_np = to_pil(output).convert("RGB")
        output_img = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
        output_img = cv2.resize(output_img, output_size)

        # Draw FPS overlay with more info
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        cv2.putText(output_img, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Avg FPS: {avg_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output_img, f"Inference: {inference_time*1000:.1f}ms", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Enhanced Output", output_img)

        # Save to file
        out.write(output_img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: Add small delay to prevent overwhelming the system
        # time.sleep(0.001)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final statistics
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    max_fps = max(fps_list)
    min_fps = min(fps_list)
    print(f"\n=== Performance Statistics ===")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Max FPS: {max_fps:.2f}")
    print(f"Min FPS: {min_fps:.2f}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed_time:.2f}s")
