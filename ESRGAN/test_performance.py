# File: ESRGAN/test_performance.py
# Performance testing script for video enhancement

import torch
import time
import numpy as np
from student_model_enhanced import load_student_model
from performance_config import PerformanceConfig
from torchvision.transforms import ToTensor
import argparse

def benchmark_model(model, input_size, num_runs=100, device='cuda'):
    """Benchmark model performance with synthetic data"""
    
    # Create synthetic input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warmup runs
    print("Warming up model...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark runs
    print(f"Running {num_runs} inference iterations...")
    times = []
    
    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.time()
            output = model(dummy_input)
            end_time = time.time()
            
            inference_time = end_time - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_runs} iterations")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'fps': fps,
        'times': times
    }

def test_configuration(target='fast', num_runs=100):
    """Test a specific performance configuration"""
    
    # Get configuration
    config = PerformanceConfig.get_config(target)
    input_size = config['resolution']['input_size']
    fps_target = config['fps_target']
    
    print(f"\n=== Testing {target.upper()} Configuration ===")
    print(f"Target FPS: {fps_target}")
    print(f"Input resolution: {input_size}")
    print(f"Skip sharpening: {config['resolution']['skip_sharpening']}")
    print(f"Mixed precision: {config['resolution']['use_mixed_precision']}")
    
    # Setup device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model with optimizations
    model = load_student_model('checkpoints/best_student_model.pth', device=device).eval()
    
    # Apply CUDA optimizations
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision('high')
    
    # Run benchmark
    results = benchmark_model(model, input_size, num_runs, device)
    
    # Print results
    print(f"\n=== Benchmark Results ===")
    print(f"Average inference time: {results['avg_time']*1000:.2f} ms")
    print(f"Standard deviation: {results['std_time']*1000:.2f} ms")
    print(f"Min time: {results['min_time']*1000:.2f} ms")
    print(f"Max time: {results['max_time']*1000:.2f} ms")
    print(f"Achieved FPS: {results['fps']:.2f}")
    print(f"Target FPS: {fps_target}")
    print(f"Target achieved: {'✅ YES' if results['fps'] >= fps_target else '❌ NO'}")
    
    return results

def compare_all_configurations(num_runs=50):
    """Compare all performance configurations"""
    
    print("=== Performance Configuration Comparison ===")
    print(f"Running {num_runs} iterations per configuration")
    
    results = {}
    
    for target in PerformanceConfig.list_targets():
        try:
            result = test_configuration(target, num_runs)
            results[target] = result
        except Exception as e:
            print(f"Error testing {target}: {e}")
            results[target] = None
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{'Configuration':<15} {'FPS':<10} {'Time (ms)':<12} {'Target':<10} {'Status':<10}")
    print(f"{'='*80}")
    
    for target, result in results.items():
        if result is not None:
            config = PerformanceConfig.get_config(target)
            fps_target = config['fps_target']
            status = "✅ PASS" if result['fps'] >= fps_target else "❌ FAIL"
            
            print(f"{target:<15} {result['fps']:<10.2f} {result['avg_time']*1000:<12.2f} "
                  f"{fps_target:<10} {status:<10}")
        else:
            print(f"{target:<15} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'ERROR':<10}")
    
    print(f"{'='*80}")
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_target = max(valid_results.keys(), key=lambda x: valid_results[x]['fps'])
        best_fps = valid_results[best_target]['fps']
        print(f"\n🏆 Best performing configuration: {best_target} ({best_fps:.2f} FPS)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Performance testing for video enhancement')
    parser.add_argument('--target', type=str, default=None,
                       choices=PerformanceConfig.list_targets(),
                       help='Test specific configuration')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all configurations')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of benchmark runs')
    parser.add_argument('--list-targets', action='store_true',
                       help='List all available targets')
    
    args = parser.parse_args()
    
    if args.list_targets:
        print("Available performance targets:")
        for target in PerformanceConfig.list_targets():
            config = PerformanceConfig.get_config(target)
            print(f"  {target}: {config['fps_target']} FPS target")
        return
    
    if args.compare_all:
        compare_all_configurations(args.runs)
    elif args.target:
        test_configuration(args.target, args.runs)
    else:
        print("Please specify --target or --compare-all")
        print("Use --help for more options")

if __name__ == "__main__":
    main() 