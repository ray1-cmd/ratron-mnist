import torch
import torch.nn as nn
from joblib import load
from huggingface_hub import hf_hub_download

# Definition of the Neural Network class for loading.
class SimpleFFNN(nn.Module):
    def __init__(self):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hugging Face repository ID
repo_id = "RAYAuser/ratron-minst-2tech"

# Downloading the files
ffnn_path = hf_hub_download(repo_id=repo_id, filename="ffnn_model_state.pt")
rf_path = hf_hub_download(repo_id=repo_id, filename="random_forest_model.joblib")

# Loading the models
ffnn_model_loaded = SimpleFFNN()
ffnn_model_loaded.load_state_dict(torch.load(ffnn_path))
ffnn_model_loaded.eval()

rf_model_loaded = load(rf_path)

print("The models have been loaded successfully.")

Prediction on New Data
Once the models are loaded, you can use them to make inferences on new images.

import numpy as np

# Creating a numpy array to simulate an image (28x28 pixels)
sample_image = np.random.rand(28, 28)

# Reshaping the input data for the models
sample_image_flat = sample_image.reshape(1, -1)
ffnn_input_tensor = torch.from_numpy(sample_image_flat).float()

# Prediction with the Neural Network
with torch.no_grad():
    output = ffnn_model_loaded(ffnn_input_tensor)
    _, ffnn_prediction = torch.max(output.data, 1)

# Prediction with the Random Forest
rf_prediction = rf_model_loaded.predict(sample_image_flat)

print(f"Prediction by the Neural Network: {ffnn_prediction.item()}")
print(f"Prediction by the Random Forest: {rf_prediction[0]}")
