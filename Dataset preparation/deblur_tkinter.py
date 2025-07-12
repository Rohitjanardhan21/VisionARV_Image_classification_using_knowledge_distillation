import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance
import torch
import time
import os, sys
from torchvision import transforms
from pytorch_msssim import ssim
import torch.nn.functional as F

# Add ESRGAN to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ESRGAN'))
from ESRGAN.student_model_enhanced import StudentNetEnhanced

class DeblurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Deblurring App - HD")
        self.root.geometry("1000x700")  # Increased window size for HD display
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        try:
            self.model = StudentNetEnhanced(channels=32, num_blocks=6)  # Match the checkpoint parameters
            self.model.load_state_dict(torch.load('checkpoints/best_student_model.pth', map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return
        
        # Transform with Ultra HD resolution for maximum clarity
        self.transform = transforms.Compose([
            transforms.Resize((768, 768)),  # Ultra HD resolution (increased from 512x512)
            transforms.ToTensor()
        ])
        
        # Enhanced sharpening function
        def enhance_sharpness(tensor, strength=0.3):
            """Enhanced sharpening with edge detection"""
            # Unsharp masking
            kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=tensor.device) / 16.0
            kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            blurred = F.conv2d(tensor.unsqueeze(0), kernel, padding=1, groups=3)
            
            # Edge enhancement
            edge_kernel = torch.tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype=torch.float32, device=tensor.device) / 8.0
            edge_kernel = edge_kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            edges = F.conv2d(tensor.unsqueeze(0), edge_kernel, padding=1, groups=3)
            
            # Combine sharpening and edge enhancement
            enhanced = tensor + strength * (tensor - blurred.squeeze(0)) + 0.1 * edges.squeeze(0)
            return torch.clamp(enhanced, 0.0, 1.0)
        
        self.enhance_sharpness = enhance_sharpness
        
        # Variables
        self.input_image = None
        self.output_image = None
        self.input_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Image Deblurring App", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Device info
        device_label = ttk.Label(main_frame, text=f"Device: {self.device.upper()}")
        device_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 20))
        
        # Select image button
        self.select_btn = ttk.Button(button_frame, text="Select Blurry Image", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        self.process_btn = ttk.Button(button_frame, text="Process Image", command=self.process_image, state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Save button
        self.save_btn = ttk.Button(button_frame, text="Save Output", command=self.save_output, state='disabled')
        self.save_btn.pack(side=tk.LEFT)
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Input image
        input_frame = ttk.LabelFrame(images_frame, text="Input (Blurry)", padding="10")
        input_frame.grid(row=0, column=0, padx=(0, 10))
        
        self.input_label = ttk.Label(input_frame, text="No image selected", width=30)
        self.input_label.pack()
        
        # Output image
        output_frame = ttk.LabelFrame(images_frame, text="Output (Sharpened)", padding="10")
        output_frame.grid(row=0, column=1)
        
        self.output_label = ttk.Label(output_frame, text="No output yet", width=30)
        self.output_label.pack()
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(main_frame, text="Metrics", padding="10")
        metrics_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # SSIM
        self.ssim_label = ttk.Label(metrics_frame, text="SSIM: --")
        self.ssim_label.grid(row=0, column=0, padx=(0, 20))
        
        # FPS
        self.fps_label = ttk.Label(metrics_frame, text="FPS: --")
        self.fps_label.grid(row=0, column=1)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Blurry Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.input_path = file_path
                self.input_image = Image.open(file_path).convert('RGB')
                
                # Resize for display (larger for HD preview)
                display_size = (300, 300)  # Increased from 200x200
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
            self.status_var.set("Processing image...")
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
            fps = 1 / (end_time - start_time)
            ssim_score = ssim(output_tensor.float(), input_tensor.float(), data_range=1.0).item()
            
            # Apply enhanced sharpening for crystal clear output
            output_tensor = output_tensor.squeeze(0).clamp(0, 1)
            output_tensor = self.enhance_sharpness(output_tensor, strength=0.4)  # Stronger sharpening
            
            # Convert to PIL image (full HD resolution)
            self.output_image = transforms.ToPILImage()(output_tensor.cpu())
            # Note: self.output_image is now 512x512 (HD resolution) with enhanced sharpness
            
            # Display output (larger for HD preview)
            display_size = (300, 300)  # Increased from 200x200
            display_output = self.output_image.copy()
            display_output.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(display_output)
            self.output_label.configure(image=photo, text="")
            self.output_label.image = photo
            
            # Update metrics
            self.ssim_label.config(text=f"SSIM: {ssim_score:.4f}")
            self.fps_label.config(text=f"FPS: {fps:.2f}")
            
            # Enable save button
            self.save_btn.config(state='normal')
            
            self.status_var.set("Processing complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            self.status_var.set("Processing failed")
    
    def save_output(self):
        if self.output_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Sharpened Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.output_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved to: {file_path}")
                self.status_var.set(f"Image saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def main():
    root = tk.Tk()
    app = DeblurApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 