import torch
import torch.nn as nn
from torchvision import models
import time

def run_benchmark(batch_size, device):
  # Check for CUDA availability
#   device = "cuda" if torch.cuda.is_available() else "cpu"
  torch.cuda.empty_cache()

  # Define model with pre-trained weights (requires internet connection)
  model = models.resnet50(pretrained=False)
  model.eval()  # Set model to evaluation mode

  # Move model to chosen device
  model.to(device)

  dataSize = 8*224*224*3*batch_size/1000000000
  print(f"Input data is about {dataSize:.3f} GB")

  # Generate random input data on chosen device
  data = torch.randn(batch_size, 3, 224, 224).to(device)  # Random image data

  # Warmup run
  model(data)

  # Start time measurement
  torch.cuda.synchronize()  # Ensure GPU operations finish before timing
  start_time = time.time()

  # Run 1000 inferences
  runs=500
  for _ in range(runs):
    model(data)

  # End time measurement
  torch.cuda.synchronize()
  end_time = time.time()

  # Calculate and print average inference time
  total_time = end_time - start_time
  average_time = total_time / runs
  print(f"Batch size {batch_size} on {device}: Average inference time - {average_time:.4f} seconds")

# Run benchmark for different batch sizes
# batch_sizes = [1, 4, 8, 16, 32, 64, 110]
batch_sizes = [64]
for batch_size in batch_sizes:
    # run_benchmark(batch_size, 'cuda')
    run_benchmark(batch_size, 'cpu')
