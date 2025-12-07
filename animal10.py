import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from training_functions import activation_function
import torch.nn as nn


def animals10_preprocessing(data_path, test_size=0.2, val_size=0.2, image_size=256, random_state=1, subset=1) -> tuple:
    """
    Preprocess images from the Animals10 dataset.
    
    Args:
        data_path: Path to the raw-img directory containing class folders
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation 
        image_size: Target size for images
        random_state: Random seed for reproducibility
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),    
        transforms.RandomRotation(degrees=15),     
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_paths = []
    labels = []
    class_names = []
    
    for class_name in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            # Get all images in class folder
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if subset < 1:
                        if np.random.rand() > subset:
                            continue
                    image_paths.append(os.path.join(class_path, filename))
                    labels.append(class_name)
    
    unique_classes = sorted(set(class_names))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    label_indices = [class_to_idx[label] for label in labels]
    

    assert test_size + val_size < 1, "test_size + val_size must be less than 1"
    

    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, label_indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=label_indices
    ) 
    

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Create datasets
    train_dataset = Animals10Dataset(X_train, y_train, transform=train_transform)
    val_dataset = Animals10Dataset(X_val, y_val, transform=val_test_transform)
    test_dataset = Animals10Dataset(X_test, y_test, transform=val_test_transform)
    

    
    return train_dataset, val_dataset, test_dataset, class_to_idx, idx_to_class


class Animals10Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (256, 256), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label
    
    def count_features(self, image_size=256):
        return 3 * image_size * image_size

class Animals10_model(torch.nn.Module):
    def __init__(self, num_classes, mode):
        self.mode = mode
        super(Animals10_model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_function(mode),     
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_function(mode), 
            nn.MaxPool2d(2, 2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_function(mode), 
            nn.MaxPool2d(2, 2)
        )
        
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
                
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), 
            activation_function(mode),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.global_pool(x)
            x = self.classifier(x)
            return x


def get_num_classes(data_path):
    class_folders = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    return len(class_folders)

