import torch

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU instead")

num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")