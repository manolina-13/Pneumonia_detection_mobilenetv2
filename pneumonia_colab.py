# ============================================================
# MobileNetV2 - Pneumonia vs Normal Image Classification
# Designed for Google Colab
# ============================================================


# ============================================================
# STEP 1: Mount your Google Drive
# (Your dataset zip file should be uploaded to Google Drive)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')


# ============================================================
# STEP 2: Unzip the dataset from Google Drive into Colab
# Change the path below to where you uploaded archive.zip
# ============================================================
import zipfile
import os

zip_path = '/content/drive/MyDrive/archive.zip'   # <-- Change this if needed
extract_path = '/content/chest_xray'

if not os.path.exists(extract_path):
    print("Unzipping dataset... this may take a minute.")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Done unzipping!")
else:
    print("Dataset already unzipped. Skipping.")


# ============================================================
# STEP 3: Import all required libraries
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


# ============================================================
# STEP 4: Configuration - change these values as needed
# ============================================================

# Path to the folder that contains 'train', 'val', and 'test' subfolders
DATA_DIR = '/content/chest_xray_data/chest_xray/chest_xray'

BATCH_SIZE = 32      # How many images to process at once
EPOCHS = 5           # How many full passes through the training data
LEARNING_RATE = 0.001  # How fast the model learns (small = stable)


# ============================================================
# STEP 5: Use GPU if available (Colab gives you a free GPU!)
# In Colab: Runtime > Change runtime type > GPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# STEP 6: Define image transforms
# All images must be resized and converted to tensors.
# We also normalize using ImageNet mean/std since MobileNetV2
# was originally trained on ImageNet images.
# For training, we also add random flips to prevent overfitting.
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),        # MobileNetV2 needs 224x224
    transforms.RandomHorizontalFlip(),    # Randomly flip image (data augmentation)
    transforms.ToTensor(),                # Convert image pixels to a PyTorch tensor
    transforms.Normalize(                 # Normalize pixel values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# STEP 7: Load the datasets using ImageFolder
# ImageFolder automatically assigns labels based on folder names.
# NORMAL = 0, PNEUMONIA = 1 (alphabetical order)
# ============================================================
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  transform=val_test_transform)

# DataLoaders break the dataset into batches and shuffle training data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training images  : {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images      : {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")  # ['NORMAL', 'PNEUMONIA']


# ============================================================
# STEP 8: Load the pre-trained MobileNetV2 model
# Transfer Learning: MobileNetV2 was already trained on millions
# of images. We reuse those learned features and only replace
# the final layer to output 2 classes (Normal vs Pneumonia).
# ============================================================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze all the base layers (feature extractor) of MobileNetV2
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final classification layer (originally 1000 classes)
# with a new layer that outputs just 2 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 2)

# Move the model to GPU (if available)
model = model.to(device)


# ============================================================
# STEP 9: Define Loss Function and Optimizer
# CrossEntropyLoss: Standard loss function for classification.
# SGD: Stochastic Gradient Descent optimizer.
# ============================================================
criterion = nn.CrossEntropyLoss()
# Ensure the optimizer ONLY updates the parameters of the final classifier layer
optimizer = optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE, momentum=0.9)

# Cosine Annealing Scheduler gently drops the learning rate following a cosine curve
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ============================================================
# STEP 10: The Training Loop
# For each epoch:
#   - Train on training data (weights ARE updated)
#   - Validate on validation data (weights are NOT updated)
# ============================================================
train_losses = []  # Store training loss per epoch for plotting
val_losses   = []  # Store validation loss per epoch for plotting

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1} / {EPOCHS} ---")

    # ---- TRAINING PHASE ----
    model.train()  # Switch model to training mode
    running_train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)   # Move images to GPU
        labels = labels.to(device)   # Move labels to GPU

        optimizer.zero_grad()        # Clear old gradients
        outputs = model(images)      # Forward pass: get predictions
        loss = criterion(outputs, labels)  # Calculate training loss
        loss.backward()              # Backpropagation: calculate gradients
        optimizer.step()             # Update the weights

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ---- VALIDATION PHASE ----
    model.eval()  # Switch model to evaluation mode (no dropout, etc.)
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation (saves memory)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)              # Forward pass only
            loss = criterion(outputs, labels)    # Calculate validation loss
            running_val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = running_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    val_accuracy = 100 * correct / total

    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val Loss   : {avg_val_loss:.4f}")
    print(f"  Val Accuracy: {val_accuracy:.2f}%")

    # Step the scheduler at the end of each epoch to update the learning rate
    scheduler.step()

print("\nTraining Complete!")


# ============================================================
# STEP 11: Plot Training Loss vs Validation Loss
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss',   marker='o', color='blue')
plt.plot(range(1, EPOCHS + 1), val_losses,   label='Validation Loss', marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()
print("Plot saved as loss_plot.png")


# ============================================================
# STEP 12: Final Test on the Test Dataset
# This is run ONCE at the very end to get the final accuracy.
# Weights are NOT updated here.
# ============================================================
print("\n--- Final Test on Test Dataset ---")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Final Test Accuracy: {test_accuracy:.2f}%")


# ============================================================
# STEP 13: Save the trained model weights
# ============================================================
torch.save(model.state_dict(), 'mobilenetv2_pneumonia.pth')
print("Model saved as mobilenetv2_pneumonia.pth")
