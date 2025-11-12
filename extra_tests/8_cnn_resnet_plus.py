# --- Imports ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys

# --- Dataset ---
class SatelliteWealthDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        with rasterio.open(img_path) as src:
            img = src.read()
            img = img.astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-7)  # Normalize

        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        label = self.dataframe.iloc[idx]['wealthpooled']
        return img, torch.tensor(label, dtype=torch.float32)

# --- Transformations ---
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=30),
    T.Resize((224, 224)),
])

test_transform = T.Compose([
    T.Resize((224, 224)),
])

# --- Functions ---
def parse_filename(filepath):
    base = os.path.basename(filepath).replace('.tif', '')
    lat, lon, year = base.split('_')
    return float(lat), float(lon), int(year)

def match_images_to_labels(csv_path, image_folder, target_column):
    dhs_df = pd.read_csv(csv_path)
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

    matches = []
    for file in image_files:
        lat, lon, year = parse_filename(file)
        distances = np.sqrt((dhs_df['LATNUM'] - lat)**2 + (dhs_df['LONGNUM'] - lon)**2)
        nearest_idx = distances.idxmin()
        wealth_value = dhs_df.loc[nearest_idx, target_column]
        matches.append({'image_path': file, 'wealthpooled': wealth_value})

    return pd.DataFrame(matches)

# --- Training ---
def train_model(model, train_loader, test_loader, device, num_epochs, save_path, use_cuda=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    scaler = torch.amp.GradScaler(enabled=use_cuda)
    # scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    train_losses = []
    test_r2s = []
    best_r2 = -float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda):
                preds = model(imgs).squeeze()
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()


        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs).squeeze().cpu().numpy()
                preds.extend(outputs)
                trues.extend(labels.numpy())

        r2 = r2_score(trues, preds)
        epoch_loss = running_loss / len(train_loader)

        # Step the scheduler based on r2
        scheduler.step(r2)

        train_losses.append(epoch_loss)
        test_r2s.append(r2)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - R²: {r2:.4f}")

        # Save the best model
        if r2 > best_r2:
            torch.save(model.state_dict(), save_path)
            best_r2 = r2

    return train_losses, test_r2s

def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

# --- Main ---
def main():
    # Paths
    CSV_FILE = os.environ.get('WS_DIR')+'/dhs_wealth_index_cleaned.csv'
    IMAGE_FOLDER = os.environ.get('WS_DIR')+'/images'
    TARGET_COLUMN = 'wealthpooled'
    SAVE_PATH = 'Resnet50_pretrained_100.pth'
    BATCH_SIZE = 128
    NUM_EPOCHS = 100

    print(CSV_FILE)
    print(IMAGE_FOLDER)

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    # Prepare dataset
    dataset_df = match_images_to_labels(CSV_FILE, IMAGE_FOLDER, TARGET_COLUMN)
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)

    train_dataset = SatelliteWealthDataset(train_df, transform=train_transform)
    test_dataset = SatelliteWealthDataset(test_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=use_cuda)

    # Load ResNet-34
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # Train
    train_losses, test_r2s = train_model(model, train_loader, test_loader, device, NUM_EPOCHS, SAVE_PATH, use_cuda)

    # After training
    smoothed_losses = smooth_curve(train_losses)
    smoothed_r2s = smooth_curve(test_r2s)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(smoothed_losses, label='Smoothed Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Smoothed)')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(smoothed_r2s, label='Smoothed Test R²', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Test R² (Smoothed)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves_smoothed.png')
    plt.show()

if __name__ == '__main__':
    main()
