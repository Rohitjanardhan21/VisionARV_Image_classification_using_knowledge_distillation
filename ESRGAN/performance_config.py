# File: ESRGAN/performance_config.py
# Performance configuration for video enhancement

class PerformanceConfig:
    """Configuration class for video enhancement performance settings"""
    
    # FPS Targets
    FPS_TARGETS = {
        'ultra_fast': 60,    # Maximum speed, lower quality
        'fast': 45,          # Balanced speed and quality
        'high_clarity': 35,  # High clarity with 30+ FPS
        'standard': 30,      # Standard performance
        'quality': 15        # Maximum quality, lower speed
    }
    
    # Resolution settings for different performance levels
    RESOLUTION_CONFIGS = {
        'ultra_fast': {
            'input_size': (128, 128),
            'output_size': (640, 480),
            'skip_sharpening': True,
            'use_mixed_precision': True,
            'buffer_size': 1
        },
        'fast': {
            'input_size': (256, 256),
            'output_size': (640, 480),
            'skip_sharpening': False,
            'use_mixed_precision': True,
            'buffer_size': 2
        },
        'high_clarity': {
            'input_size': (384, 384),
            'output_size': (640, 480),
            'skip_sharpening': False,
            'use_mixed_precision': True,
            'buffer_size': 2,
            'enhanced_processing': True
        },
        'standard': {
            'input_size': (384, 384),
            'output_size': (640, 480),
            'skip_sharpening': False,
            'use_mixed_precision': False,
            'buffer_size': 3
        },
        'quality': {
            'input_size': (512, 512),
            'output_size': (640, 480),
            'skip_sharpening': False,
            'use_mixed_precision': False,
            'buffer_size': 5
        }
    }
    
    # CUDA optimizations
    CUDA_OPTIMIZATIONS = {
        'cudnn_benchmark': True,
        'cudnn_deterministic': False,
        'cudnn_enabled': True,
        'float32_matmul_precision': 'high'
    }
    
    # Camera settings
    CAMERA_SETTINGS = {
        'width': 640,
        'height': 480,
        'fps': 60,
        'buffer_size': 1,
        'fourcc': 'MJPG'  # Use MJPG for faster capture
    }
    
    @classmethod
    def get_config(cls, target='fast'):
        """Get configuration for a specific performance target"""
        if target not in cls.FPS_TARGETS:
            raise ValueError(f"Invalid target: {target}. Available: {list(cls.FPS_TARGETS.keys())}")
        
        return {
            'fps_target': cls.FPS_TARGETS[target],
            'resolution': cls.RESOLUTION_CONFIGS[target],
            'cuda_optimizations': cls.CUDA_OPTIMIZATIONS,
            'camera_settings': cls.CAMERA_SETTINGS
        }
    
    @classmethod
    def list_targets(cls):
        """List all available performance targets"""
        return list(cls.FPS_TARGETS.keys())
    
    @classmethod
    def get_recommended_target(cls, gpu_memory_gb=8):
        """Get recommended target based on GPU memory"""
        if gpu_memory_gb >= 12:
            return 'ultra_fast'
        elif gpu_memory_gb >= 8:
            return 'fast'
        elif gpu_memory_gb >= 6:
            return 'standard'
        else:
            return 'quality'

# Example usage:
if __name__ == "__main__":
    print("Available performance targets:")
    for target in PerformanceConfig.list_targets():
        config = PerformanceConfig.get_config(target)
        print(f"  {target}: {config['fps_target']} FPS target")
        print(f"    Input resolution: {config['resolution']['input_size']}")
        print(f"    Skip sharpening: {config['resolution']['skip_sharpening']}")
        print()
    
    # Get recommended config for your system
    recommended = PerformanceConfig.get_recommended_target()
    print(f"Recommended target for your system: {recommended}") 