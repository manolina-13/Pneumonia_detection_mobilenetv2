import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# ==========================================
# 1. Configuration
# ==========================================
MODEL_WEIGHTS_PATH = 'mobilenetv2_pneumonia.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA'] # Must match the alphabetical order of your train folders

def predict_image(image_path):
    # ==========================================
    # 2. Setup Device
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 3. Load the Trained Model
    # ==========================================
    # We must recreate the exact same model architecture first
    model = models.mobilenet_v2(weights=None) # We don't need pretrained weights here, we will load our own
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    # Load our saved weights into the model
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print("Successfully loaded model weights.")
    except FileNotFoundError:
        print(f"Error: Could not find '{MODEL_WEIGHTS_PATH}'.")
        print("Make sure you have run the training script first!")
        return

    model = model.to(device)
    model.eval() # Set model to evaluation mode (important for inference)

    # ==========================================
    # 4. Prepare the Image
    # ==========================================
    # We must apply the EXACT same transformations used for the validation set during training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Transform the image and add a batch dimension (models expect batches)
    # [Channels, Height, Width] -> [1, Channels, Height, Width]
    input_tensor = transform(image).unsqueeze(0).to(device)

    # ==========================================
    # 5. Make a Prediction
    # ==========================================
    with torch.no_grad(): # We don't need to track gradients for inference
        outputs = model(input_tensor)
        
        # Get the predicted class index
        _, predicted_idx = torch.max(outputs, 1)
        
        # Get probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = probabilities[predicted_idx.item()].item() * 100

    print("\n" + "="*30)
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*30 + "\n")

if __name__ == '__main__':
    # You can pass the image path as a command line argument
    # Example: python test_pneumonia_model.py my_xray.jpeg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_image(img_path)
    else:
        print("Please provide an image path.")
        print("Usage: python test_pneumonia_model.py <path_to_image>")
