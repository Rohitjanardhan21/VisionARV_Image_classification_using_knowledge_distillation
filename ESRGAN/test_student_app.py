#!/usr/bin/env python3
"""
Test script for the Student Model Tkinter Application
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import Image, ImageTk
        import torch
        import time
        from torchvision import transforms
        from pytorch_msssim import ssim
        import torch.nn.functional as F
        
        print("✅ All basic imports successful")
        
        # Test student model import
        from student_model_enhanced import load_student_model
        print("✅ Student model import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if the student model can be loaded"""
    try:
        from student_model_enhanced import load_student_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_student_model().to(device)
        
        checkpoint_path = 'checkpoints/best_student_model.pth'
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            print(f"✅ Student model loaded successfully on {device}")
            return True
        else:
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Model created but not loaded with weights")
            return True
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    print("🧪 Testing Student Model Tkinter Application")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your dependencies.")
        return
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Model loading test failed.")
        return
    
    print("\n✅ All tests passed!")
    print("\n🎉 You can now run the application with:")
    print("   python student_visualize_tkinter.py")
    
    # Ask if user wants to run the app
    try:
        response = input("\nWould you like to run the Tkinter app now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n🚀 Starting Tkinter application...")
            from student_visualize_tkinter import main as run_app
            run_app()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 