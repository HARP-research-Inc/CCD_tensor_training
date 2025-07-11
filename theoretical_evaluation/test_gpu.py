import torch

print("\n=== PyTorch GPU Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    # Test GPU computation
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = torch.matmul(x, y.t())
    print("\nGPU computation successful!")
    print(f"Result shape: {z.shape}")
    print(f"Result device: {z.device}")
else:
    print("\nNo CUDA device available!")
print("=====================\n") 