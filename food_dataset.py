"""
Dataset Loader untuk Indonesian Food Classification
35 Classes - CSV Format dengan One-Hot Encoding
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class IndonesianFoodDatasetCSV(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file: Path ke train.csv atau test.csv
            img_dir: Directory dengan images (food-tfk-images/)
            transform: Image transformations
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Get class names (kolom 3 sampai akhir adalah class labels)
        self.class_names = list(self.df.columns[3:])
        self.num_classes = len(self.class_names)
        
        print(f"Loaded {len(self.df)} images")
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image filename
        img_name = self.df.iloc[idx, 0]  # Column 'Image Index'
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return blank image if error
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label (one-hot encoded columns)
        label_array = self.df.iloc[idx, 3:].values.astype('float32')
        label = torch.argmax(torch.tensor(label_array)).item()
        
        return image, label

def get_train_transforms():
    """Data augmentation untuk training"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        # RandomErasing helps regularize and reduce overfitting
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
    ])

def get_test_transforms():
    """Standard transforms untuk test/validation"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(data_dir='Dataset ML', batch_size=32, num_workers=0):
    """
    Create dataloaders untuk train, validation, dan test
    """
    train_dataset = IndonesianFoodDatasetCSV(
        csv_file=os.path.join(data_dir, 'train.csv'),
        img_dir=os.path.join(data_dir, 'food-tfk-images'),
        transform=get_train_transforms()
    )
    
    dev_dataset = IndonesianFoodDatasetCSV(
        csv_file=os.path.join(data_dir, 'dev.csv'),
        img_dir=os.path.join(data_dir, 'food-tfk-images'),
        transform=get_test_transforms()
    )
    
    test_dataset = IndonesianFoodDatasetCSV(
        csv_file=os.path.join(data_dir, 'test.csv'),
        img_dir=os.path.join(data_dir, 'food-tfk-images'),
        transform=get_test_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, test_loader, train_dataset.class_names

if __name__ == "__main__":
    # Test dataset loading
    print("Testing Dataset Loading...")
    print("="*60)
    
    train_loader, dev_loader, test_loader, class_names = create_dataloaders(
        data_dir='Dataset ML',
        batch_size=8
    )
    
    print(f"\nClass names ({len(class_names)}):")
    for i, name in enumerate(class_names, 1):
        print(f"{i:2d}. {name}")
    
    print(f"\nDataset sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading one batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sample labels: {labels[:5]}")
