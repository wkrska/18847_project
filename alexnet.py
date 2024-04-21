import csv
import torch
import torch.nn as nn
import torchvision.models as models
from timeit import default_timer

# Define helper function to measure execution time
def measure_time(model, input_data):
  start_time = default_timer()
  with torch.no_grad():
    _ = model(input_data)
  end_time = default_timer()
  return end_time - start_time

# Check if CUDA is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Load pre-trained AlexNet model
model = models.alexnet(pretrained=True).to(device)

with open("data.csv", "a", newline="") as csvfile:
  # Create a CSV writer object
  writer = csv.writer(csvfile, delimiter='\t')

  # Run for different image input sizes
  for img_size in [64, 128, 256, 512, 1024] :
    for batch_size in [1, 10, 50, 100, 200, 400, 600, 800, 1000]:
      # Load pre-trained AlexNet model
      model = models.alexnet(pretrained=True).to(device)
      
      try :
        # Prepare random input data
        input_shape = (3, img_size, img_size)
        input_data = torch.randn(batch_size, *input_shape).to(device)
    
        # Warm-up run
        _ = measure_time(model, input_data)
    
        # Perform multiple runs and measure average time
        num_runs = 100
        total_time = 0
        for _ in range(num_runs):
            total_time += measure_time(model, input_data)
    
        # Calculate and print average inference time
        avg_time = total_time / num_runs
        #   print(f"Average inference time for AlexNet on {device}: {avg_time:.4f} seconds")
        # Write the data to the CSV file
        writer.writerow([img_size,batch_size,avg_time])
      except : 
        print("Too big!")