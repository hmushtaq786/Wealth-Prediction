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
import optuna
import sys
import joblib
import torch.nn.functional as F
import random

# --- Dataset ---
class SatelliteWealthDataset(Dataset):
    def __init__(self, dataframe, transform=False, augment=False):
        self.dataframe = dataframe
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
            mean = img.mean()
            std = img.std()
            if std < 1e-6:  # Handle uniform or constant images
                std = 1e-6
            img = (img - mean) / std

        img = torch.tensor(img, dtype=torch.float32)  # (C, H, W)

        # Resize to 224x224
        if self.transform:
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        # Manual augmentation (optional)
        if self.augment:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])  # Horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])  # Vertical flip

        label = self.dataframe.iloc[idx]['wealthpooled']
        return img, torch.tensor(label, dtype=torch.float32)

# --- Helper functions ---
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

# --- Training function ---
def train_model(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, num_epochs, use_cuda=False):
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
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs).squeeze()
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)  # <-- Fix here
                preds.extend(outputs.cpu().numpy())
                trues.extend(labels.numpy())

        # Convert to numpy safely
        preds = np.nan_to_num(np.array(preds), nan=0.0, posinf=1e6, neginf=-1e6)
        trues = np.nan_to_num(np.array(trues), nan=0.0, posinf=1e6, neginf=-1e6)
        r2 = r2_score(trues, preds)
        epoch_loss = running_loss / len(train_loader)
        scheduler.step(r2)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - R²: {r2:.4f}")

        train_losses.append(epoch_loss)
        test_r2s.append(r2)

    return train_losses, test_r2s, max(test_r2s)

# --- Optuna objective function ---
def objective(trial):
    # Paths
    CSV_FILE = os.environ.get('WS_DIR') + '/dhs_wealth_index_cleaned.csv'
    IMAGE_FOLDER = os.environ.get('WS_DIR') + '/images'
    TARGET_COLUMN = 'wealthpooled'

    # Hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.3, 0.5)
    optimizer_choice = trial.suggest_categorical("optimizer", ["Adam"])
    num_epochs = 70

    # Transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=30),
        T.Resize((224, 224)),
    ])
    test_transform = T.Compose([T.Resize((224, 224))])

    # Prepare dataset
    dataset_df = match_images_to_labels(CSV_FILE, IMAGE_FOLDER, TARGET_COLUMN)
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    train_dataset = SatelliteWealthDataset(train_df, transform=True, augment=True)
    test_dataset = SatelliteWealthDataset(test_df, transform=True, augment=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=use_cuda)

    # Model
    print(f"VGG-16")
    print(f"Batch Size: {batch_size} - LR: {lr} - Weight Decay: {weight_decay} - Dropout Rate: {dropout_rate}")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.features[0] = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)

    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.classifier[-1].in_features, 1)
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Train model
    train_losses, test_r2s, best_r2 = train_model(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, num_epochs=num_epochs, use_cuda=use_cuda)

    # Save best model for this trial
    torch.save(model.state_dict(), f"optuna/models/vgg16/trial_{trial.number}.pth")

    # Save R² scores per epoch for this trial
    np.save(f"optuna/scores/vgg16/trial_{trial.number}_r2.npy", np.array(test_r2s))

    return best_r2

def plot_optuna_study(study, scores_path):
    # Plot R² over epochs for all trials
    plt.figure(figsize=(12, 6))
    for trial in study.trials:
        try:
            r2_scores = np.load(f"{scores_path}_{trial.number}_r2.npy")
            plt.plot(range(1, len(r2_scores) + 1), r2_scores, label=f'Trial {trial.number}')
        except FileNotFoundError:
            continue  # Skip trials that failed or didn’t save R²

    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.title("R² Score per Epoch for Each Trial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optuna/plots/vgg16.png")
    plt.show()

# --- Main ---
def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=7)

    print("\nBest trial:")
    print(study.best_trial.params)

    scores_path = "optuna/scores/vgg16/trial"
    plot_optuna_study(study, scores_path)

    # Save the Optuna study
    joblib.dump(study, "optuna/studies/vgg16.pkl")

if __name__ == '__main__':
    main()
