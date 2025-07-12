# File: ESRGAN/video_test_configurable.py
# Configurable video enhancement with performance targets

import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from student_model_enhanced import load_student_model
from performance_config import PerformanceConfig
from collections import deque
import argparse

def setup_model(device):
    """Setup model with optimizations"""
    model = load_student_model('checkpoints/best_student_model.pth', device=device).eval()
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('high')
    
    return model

def setup_camera(config):
    """Setup camera with performance settings"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    
    # Apply camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera_settings']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera_settings']['height'])
    cap.set(cv2.CAP_PROP_FPS, config['camera_settings']['fps'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config['camera_settings']['buffer_size'])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['camera_settings']['fourcc']))
    
    return cap

def preprocess_frame(frame, input_size, device):
    """Preprocess frame for model input"""
    resized = cv2.resize(frame, input_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = ToTensor()(rgb).unsqueeze(0).to(device, non_blocking=True)
    return tensor

def postprocess_frame(output_tensor, output_size):
    """Postprocess model output"""
    output = torch.clamp(output_tensor.squeeze(0), 0.0, 1.0)
    output_np = ToPILImage()(output).convert("RGB")
    output_img = cv2.cvtColor(np.array(output_np), cv2.COLOR_RGB2BGR)
    return cv2.resize(output_img, output_size, interpolation=cv2.INTER_LINEAR)

def sharpen_image_fast(tensor, strength=0.15):
    """Fast sharpening function"""
    kernel = torch.tensor([[0, 1, 0], [1, 2, 1], [0, 1, 0]],
                          dtype=torch.float32, device=tensor.device) / 6.0
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
    sharpened = tensor + strength * (tensor - blurred.squeeze(0))
    return torch.clamp(sharpened, 0.0, 1.0)

def run_video_enhancement(target='fast'):
    """Run video enhancement with specified performance target"""
    
    # Get configuration
    config = PerformanceConfig.get_config(target)
    fps_target = config['fps_target']
    resolution_config = config['resolution']
    
    print(f"=== Video Enhancement - {target.upper()} Mode ===")
    print(f"Target FPS: {fps_target}")
    print(f"Input resolution: {resolution_config['input_size']}")
    print(f"Output resolution: {resolution_config['output_size']}")
    print(f"Skip sharpening: {resolution_config['skip_sharpening']}")
    print(f"Mixed precision: {resolution_config['use_mixed_precision']}")
    
    # Setup device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = setup_model(device)
    
    # Setup camera
    cap = setup_camera(config)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = f'output_sharpened_{target}.avi'
    out = cv2.VideoWriter(output_filename, fourcc, fps_target, 
                         resolution_config['output_size'])
    
    # Performance tracking
    fps_list = deque(maxlen=60)
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    
    print("Press 'q' to quit.")
    
    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocessing
                input_tensor = preprocess_frame(frame, resolution_config['input_size'], device)
                
                # Inference timing
                inference_start = time.time()
                
                # Model inference
                output = model(input_tensor)
                
                # Optional sharpening
                if not resolution_config['skip_sharpening']:
                    output = sharpen_image_fast(output.squeeze(0), strength=0.15).unsqueeze(0)
                
                # Inference end
                inference_end = time.time()
                inference_time = inference_end - inference_start
                
                # Postprocessing
                output_img = postprocess_frame(output, resolution_config['output_size'])
                
                # FPS calculation
                frame_count += 1
                current_time = time.time()
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_list.append(current_fps)
                
                # Calculate average FPS
                if current_time - last_fps_time >= 1.0:
                    recent_fps = [f for f in fps_list if f > 0]
                    avg_fps = sum(recent_fps) / len(recent_fps) if recent_fps else 0
                    last_fps_time = current_time
                
                # Draw performance overlay
                cv2.putText(output_img, f"Mode: {target.upper()}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output_img, f"FPS: {current_fps:.1f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output_img, f"Target: {fps_target}", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output_img, f"Inference: {inference_time*1000:.1f}ms", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Show image
                cv2.imshow(f"Enhanced Output - {target.upper()}", output_img)
                
                # Save to file
                out.write(output_img)
                
                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if fps_list:
            valid_fps = [f for f in fps_list if f > 0]
            if valid_fps:
                avg_fps = sum(valid_fps) / len(valid_fps)
                max_fps = max(valid_fps)
                min_fps = min(valid_fps)
                elapsed_time = time.time() - start_time
                
                print(f"\n=== Performance Statistics ===")
                print(f"Target FPS: {fps_target}")
                print(f"Average FPS: {avg_fps:.2f}")
                print(f"Max FPS: {max_fps:.2f}")
                print(f"Min FPS: {min_fps:.2f}")
                print(f"Total frames: {frame_count}")
                print(f"Total time: {elapsed_time:.2f}s")
                print(f"Target achieved: {'✅ YES' if avg_fps >= fps_target else '❌ NO'}")
                
                if avg_fps >= fps_target:
                    print(f"🎉 Successfully achieved {avg_fps:.1f} FPS (≥{fps_target} FPS target)!")
                else:
                    print(f"⚠️  Current FPS: {avg_fps:.1f} - Consider using a lower target")

def main():
    parser = argparse.ArgumentParser(description='Configurable video enhancement')
    parser.add_argument('--target', type=str, default='fast', 
                       choices=PerformanceConfig.list_targets(),
                       help='Performance target (ultra_fast, fast, standard, quality)')
    parser.add_argument('--list-targets', action='store_true',
                       help='List all available performance targets')
    
    args = parser.parse_args()
    
    if args.list_targets:
        print("Available performance targets:")
        for target in PerformanceConfig.list_targets():
            config = PerformanceConfig.get_config(target)
            print(f"  {target}: {config['fps_target']} FPS target")
            print(f"    Input resolution: {config['resolution']['input_size']}")
            print(f"    Skip sharpening: {config['resolution']['skip_sharpening']}")
            print()
        return
    
    run_video_enhancement(args.target)

if __name__ == "__main__":
    main() 