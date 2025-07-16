import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import os
import time
import copy
import numpy as np

# --- Configuration ---
# The dataset is expected to be in a folder named 'chest_xray'
# in the same directory as this script.
DATA_DIR = './chest_xray/'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25 # Max epochs, but Early Stopping will likely stop it sooner
NUM_CLASSES = 2 # NORMAL vs PNEUMONIA
MODEL_SAVE_PATH = 'pneumoscan_efficientnet.pth'

# --- Data Loading and Preprocessing ---
# Define transformations. Note the more aggressive augmentation for training.
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Model Definition and Training Loop ---
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, patience=5):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # For Early Stopping
    epochs_no_improve = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # Get total number of batches for progress tracking
            num_batches = len(dataloaders[phase])

            # Iterate over data with progress indicator
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # Print progress every 10 batches
                if (i + 1) % 10 == 0:
                    print(f'\r{phase.capitalize()} Phase: Batch {i+1}/{num_batches}', end='')

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            # Clear the progress line before printing summary
            print('\r' + ' ' * 50, end='\r')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs.')
            break
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# This is the main execution block
if __name__ == '__main__':
    print("Initializing Datasets and Dataloaders...")

    # Create datasets for training, validation, and testing
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val']),
        'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
    }

    # Create dataloaders
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Class names: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Handle Data Imbalance with Class Weights
    num_normal = len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')))
    num_pneumonia = len(os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')))
    total_train = num_normal + num_pneumonia

    weight_for_normal = total_train / (2.0 * num_normal)
    weight_for_pneumonia = total_train / (2.0 * num_pneumonia)

    class_weights = torch.FloatTensor([weight_for_normal, weight_for_pneumonia]).to(device)
    print(f"Class weights: NORMAL={weight_for_normal:.2f}, PNEUMONIA={weight_for_pneumonia:.2f}")

    # Load pre-trained EfficientNetB0
    model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze pre-trained layers
    for param in model_ft.parameters():
        param.requires_grad = False

    # Replace the classifier
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    model_ft = model_ft.to(device)

    # Use the calculated class weights in the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Observe that only parameters of the final layer are being optimized
    optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=0.001)

    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("Starting model training...")
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs=EPOCHS, patience=5)

    # Save the Trained Model
    print(f"\nSaving model to {MODEL_SAVE_PATH}")
    torch.save({
        'model_state_dict': model_ft.state_dict(),
        'class_names': class_names
    }, MODEL_SAVE_PATH)
    print("Model and class names saved successfully.")

    # Final Evaluation on Test Set
    print("\nEvaluating on the test set...")
    model_ft.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Accuracy: {test_acc:.4f}')
    print("Training and evaluation complete.")