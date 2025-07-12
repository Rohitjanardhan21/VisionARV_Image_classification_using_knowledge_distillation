import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import time
import os
import sys
from torchvision import transforms
from pytorch_msssim import ssim
import torch.nn.functional as F

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from student_model_enhanced import load_student_model

class StudentVisualizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Model Image Deblurring App")
        self.root.geometry("1200x800")
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load student model
        try:
            self.model = load_student_model().to(self.device)
            checkpoint_path = 'checkpoints/best_student_model.pth'
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.eval()
                print(f"Student model loaded successfully on {self.device}")
            else:
                messagebox.showerror("Error", f"Checkpoint not found: {checkpoint_path}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load student model: {str(e)}")
            return
        
        # Sharpening function for enhanced output (copied from visualize_student_output.py)
        def sharpen_image(tensor, strength=0.4):
            """Apply unsharp masking to enhance details"""
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=tensor.device) / 16.0
            kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
            sharpened = tensor + strength * (tensor - blurred.squeeze(0))
            return torch.clamp(sharpened, 0.0, 1.0)
        self.sharpen_image = sharpen_image

        # Transform for processing (no normalization, matches training)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
        # Variables
        self.input_image = None
        self.output_image = None
        self.input_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Student Model Image Deblurring", 
                               font=("Arial", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Device info
        device_label = ttk.Label(main_frame, text=f"Device: {self.device.upper()}", 
                                font=("Arial", 10))
        device_label.grid(row=1, column=0, columnspan=3, pady=(0, 15))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(0, 20))
        
        # Select image button
        self.select_btn = ttk.Button(button_frame, text="Select Blurry Image", 
                                    command=self.select_image, style='Accent.TButton')
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_btn = ttk.Button(button_frame, text="Process with Student Model", 
                                     command=self.process_image, state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save Deblurred Image", 
                                  command=self.save_output, state='disabled')
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        self.clear_btn = ttk.Button(button_frame, text="Clear", 
                                   command=self.clear_images)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=3, column=0, columnspan=3, pady=(0, 20), sticky="ew")
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        
        # Input image
        input_frame = ttk.LabelFrame(images_frame, text="Input (Blurry Image)", padding="10")
        input_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        
        self.input_label = ttk.Label(input_frame, text="No image selected", 
                                    width=40, anchor='center')
        self.input_label.pack(expand=True, fill='both')
        
        # Output image
        output_frame = ttk.LabelFrame(images_frame, text="Output (Deblurred by Student Model)", padding="10")
        output_frame.grid(row=0, column=1, sticky="nsew")
        
        self.output_label = ttk.Label(output_frame, text="No output yet", 
                                     width=40, anchor='center')
        self.output_label.pack(expand=True, fill='both')
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Performance Metrics", padding="15")
        metrics_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 15))
        
        # SSIM
        self.ssim_label = ttk.Label(metrics_frame, text="SSIM: --", font=("Arial", 12))
        self.ssim_label.grid(row=0, column=0, padx=(0, 30))
        
        # FPS
        self.fps_label = ttk.Label(metrics_frame, text="Processing Speed: -- FPS", font=("Arial", 12))
        self.fps_label.grid(row=0, column=1, padx=(0, 30))
        
        # Processing time
        self.time_label = ttk.Label(metrics_frame, text="Processing Time: -- ms", font=("Arial", 12))
        self.time_label.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select a blurry image to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Blurry Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        
        if file_path:
            try:
                self.input_path = file_path
                self.input_image = Image.open(file_path).convert('RGB')
                
                # Resize for display
                display_size = (350, 350)
                display_image = self.input_image.copy()
                display_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(display_image)
                self.input_label.configure(image=photo, text="")
                self.input_label.image = photo  # Keep a reference
                
                self.process_btn.config(state='normal')
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def process_image(self):
        if self.input_image is None:
            return
        
        try:
            self.status_var.set("Processing image with student model...")
            self.root.update()
            
            # Prepare input
            input_tensor = self.transform(self.input_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                start_time = time.time()
                output_tensor = self.model(input_tensor)
                torch.cuda.synchronize() if self.device == 'cuda' else None
                end_time = time.time()
            
            # Calculate metrics
            processing_time = (end_time - start_time) * 1000  # ms
            fps = 1 / (end_time - start_time)
            
            # Clamp and sharpen (order matches script)
            output_tensor = output_tensor.squeeze(0)
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
            output_tensor = self.sharpen_image(output_tensor, strength=0.2)
            
            # Convert to PIL image (full 512x512)
            self.output_image = transforms.ToPILImage()(output_tensor.cpu())
            
            # Display output (full 512x512, no thumbnail)
            photo = ImageTk.PhotoImage(self.output_image)
            self.output_label.configure(image=photo, text="")
            self.output_label.image = photo
            
            # Update metrics
            self.ssim_label.config(text=f"SSIM: {ssim(output_tensor.unsqueeze(0), input_tensor, data_range=1.0).item():.4f}")
            self.fps_label.config(text=f"Processing Speed: {fps:.2f} FPS")
            self.time_label.config(text=f"Processing Time: {processing_time:.1f} ms")
            
            # Enable save button
            self.save_btn.config(state='normal')
            
            self.status_var.set("Processing complete! Student model has deblurred the image.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_var.set("Processing failed")
    
    def save_output(self):
        if self.output_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Deblurred Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.output_image.save(file_path)
                messagebox.showinfo("Success", f"Deblurred image saved to: {file_path}")
                self.status_var.set(f"Image saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def clear_images(self):
        """Clear both input and output images"""
        self.input_image = None
        self.output_image = None
        self.input_path = None
        
        # Reset labels
        self.input_label.configure(image="", text="No image selected")
        self.output_label.configure(image="", text="No output yet")
        
        # Reset buttons
        self.process_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        # Reset metrics
        self.ssim_label.config(text="SSIM: --")
        self.fps_label.config(text="Processing Speed: -- FPS")
        self.time_label.config(text="Processing Time: -- ms")
        
        self.status_var.set("Ready - Select a blurry image to begin")

def main():
    root = tk.Tk()
    app = StudentVisualizeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 