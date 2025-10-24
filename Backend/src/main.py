from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS for cross-origin requests
import os
import torch
import numpy as np
import io
from PIL import Image
import torchvision.transforms as transforms

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Define the colorization model
class ColorizationNet(torch.nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=4, dilation=2)
        self.conv4 = torch.nn.Conv2d(128, 3, kernel_size=5, stride=1, padding=4, dilation=2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

# Load the PyTorch model
model = ColorizationNet()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Make sure the model is on the CPU
model.eval()

# Preprocessing function to convert RGB to grayscale
def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    # You can normalize if needed: transforms.Normalize(mean=[0.5], std=[0.5])
])

# Helper function to colorize an image using the model
def colorize_image(img_path):
    # Load the image
    img = Image.open(img_path).convert('RGB')

    # Convert the image to grayscale
    gray_img = img.convert('L')

    # Apply the transformations
    img_tensor = transform(gray_img).unsqueeze(0)  # Add batch dimension

    # Ensure the model is in evaluation mode
    model.eval()

    # Get the model's output on CPU
    with torch.no_grad():  # Disable gradient calculation during inference
        colorized_tensor = model(img_tensor)

    # Convert the tensor back to an image
    colorized_img = transforms.ToPILImage()(colorized_tensor.squeeze(0))  # Remove batch dimension
    return colorized_img

# Route for the colorization endpoint
@app.route('/colorize', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Colorize the image
    colorized_image = colorize_image(file_path)

    # Convert the colorized image to a BytesIO object
    buf = io.BytesIO()
    colorized_image.save(buf, format="JPEG")
    buf.seek(0)

    # Return the colorized image
    return send_file(buf, mimetype='image/jpeg')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
