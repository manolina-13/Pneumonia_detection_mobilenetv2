import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
# IMPORTANT: Change this path to where your dataset is stored!
# The dataset should have 'train' and 'val' folders, 
# and inside each, 'NORMAL' and 'PNEUMONIA' folders.
DATA_DIR = r'c:\Users\manol\Desktop\Project\archive\chest_xray' 
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

def main():
    # ==========================================
    # 2. Setup Device (Use GPU if available)
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 3. Data Augmentation and Transforms
    # ==========================================
    # Deep learning models need image data to be tensors and normalized.
    # We add variations (augmentations) to training data to prevent overfitting.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # MobileNetV2 expects 224x224 images
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean & std
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ==========================================
    # 4. Load Dataset using ImageFolder
    # ==========================================
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found!")
        print("Please set DATA_DIR to the folder containing 'train' and 'val' subfolders.")
        return

    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), data_transforms['val'])
    }
    
    # DataLoaders handle batching and shuffling of the data
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
        'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Found classes: {class_names}")

    # ==========================================
    # 5. Initialize MobileNetV2 Model
    # ==========================================
    # Load a pretrained MobileNetV2 (trained on millions of generic images)
    # Note: 'pretrained=True' is deprecated in newer versions, use 'weights=models.MobileNet_V2_Weights.DEFAULT'
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Modify the final classification layer for our 2 classes (NORMAL, PNEUMONIA)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(device)

    # ==========================================
    # 6. Loss Function and Optimizer
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Variables to track loss for plotting
    train_losses = []
    val_losses = []

    # ==========================================
    # 7. Training Loop
    # ==========================================
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the batches of data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset gradients before the forward pass
                optimizer.zero_grad()

                # Forward pass - track history only if in 'train'
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass & update weights only if in 'train' phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate average loss and accuracy for this epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save losses for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)

    print('\nTraining complete!')

    # ==========================================
    # 8. Plot Training vs Validation Loss
    # ==========================================
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to an image file and show it
    plt.savefig('loss_plot_pneumonia.png')
    plt.show()

    # ==========================================
    # 9. Save the Trained Model
    # ==========================================
    torch.save(model.state_dict(), 'mobilenetv2_pneumonia.pth')
    print("Model weights saved as 'mobilenetv2_pneumonia.pth'")
    print("Loss plot saved as 'loss_plot_pneumonia.png'")

if __name__ == '__main__':
    main()
