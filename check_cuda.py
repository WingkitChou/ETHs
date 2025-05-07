import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version by PyTorch: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    try:
        print("\nAttempting a simple CUDA operation...")
        # Create a tensor on CPU
        x_cpu = torch.tensor([1.0, 2.0, 3.0])
        print(f"  x_cpu: {x_cpu}, device: {x_cpu.device}")
        
        # Move to GPU
        device = torch.device("cuda")
        x_gpu = x_cpu.to(device)
        print(f"  x_gpu: {x_gpu}, device: {x_gpu.device}")
        
        # Perform a simple operation on GPU
        y_gpu = x_gpu * 2
        print(f"  y_gpu (x_gpu * 2): {y_gpu}, device: {y_gpu.device}")
        
        # Move back to CPU
        y_cpu = y_gpu.to("cpu")
        print(f"  y_cpu (from y_gpu): {y_cpu}, device: {y_cpu.device}")
        print("Simple CUDA operation successful!")
        
    except Exception as e:
        print(f"Error during CUDA operation: {e}")
else:
    print("CUDA is not available. Please check your PyTorch installation and CUDA drivers.")