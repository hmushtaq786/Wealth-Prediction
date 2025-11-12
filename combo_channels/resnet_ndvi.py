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
            img = src.read(1)  # Read the first band only (1-channel)
            img = img.astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-7)  # Normalize
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dim: (1, H, W)

        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        label = self.dataframe.iloc[idx]['wealthpooled']
        return img, torch.tensor(label, dtype=torch.float32)

# --- Helper functions ---
def parse_filename(filepath):
    base = os.path.basename(filepath).replace('.tif', '')
    index, lat, lon, year = base.split('_')
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
        scheduler.step(r2)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - R²: {r2:.4f}")

        train_losses.append(epoch_loss)
        test_r2s.append(r2)

    return train_losses, test_r2s, max(test_r2s)

# --- Optuna objective function ---
def objective(trial):
    # Paths
    CSV_FILE = os.environ.get('WS_DIR') + '/dhs_wealth_index_cleaned.csv'
    IMAGE_FOLDER = os.environ.get('WS_DIR') + '/processed/NDVI/ndvi_images'
    TARGET_COLUMN = 'wealthpooled'

    # Hyperparameters to tune
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
    optimizer_choice = trial.suggest_categorical("optimizer", ["Adam"])
    num_epochs = 80

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
    train_dataset = SatelliteWealthDataset(train_df, transform=train_transform)
    test_dataset = SatelliteWealthDataset(test_df, transform=test_transform)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=use_cuda)

    # Model
    print(f"ResNet-34")
    print(f"Batch Size: {batch_size} - LR: {lr} - Weight Decay: {weight_decay} - Dropout Rate: {dropout_rate}")
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, 1)
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Train model
    train_losses, test_r2s, best_r2 = train_model(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, num_epochs=num_epochs, use_cuda=use_cuda)

    TRIAL_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", trial.number))  # Fallback if running locally

    # Save best model for this trial
    torch.save(model.state_dict(), f"../optuna/models/resnet34/trial_{TRIAL_ID}_ndvi.pth")

    # Save R² scores per epoch for this trial
    print(trial)
    np.save(f"../optuna/scores/resnet34/trial_{TRIAL_ID}_r2_ndvi.npy", np.array(test_r2s))

    return best_r2

def smooth_curve(values, window=3):
    return np.convolve(values, np.ones(window)/window, mode='valid')

def plot_optuna_study(study, scores_path):
    # Plot R² over epochs for all trials
    plt.figure(figsize=(12, 6))
    for trial in study.trials:
        trial_id = trial.user_attrs.get("slurm_id", trial.number)
        try:
            r2_scores = np.load(f"{scores_path}_{trial_id}_r2_ndvi.npy")
            r2_scores = np.clip(r2_scores, -1, 1)  # Prevent outlier distortion
            smoothed = smooth_curve(r2_scores, window=2)  # Apply smoothing
            if np.mean(r2_scores) > 0:  # "successful" trial
                plt.plot(smoothed, alpha=0.9, label=f'Trial {trial_id}')
            else:  # faded failed trials
                plt.plot(smoothed, alpha=0.3, linestyle="--")
            # plt.plot(range(1, len(smoothed) + 1), smoothed, label=f'Trial {trial_id}')
        except FileNotFoundError:
            print(f"Trial {trial_id} .npy file not found.")
            continue  # Skip trials that failed or didn’t save R²

    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.title("R² Score per Epoch for Each Trial")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("optuna/plots/resnet34_ndvi.png")
    plt.close()


def all_trials_completed(study, n_trials_expected):
    completed = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    return completed >= n_trials_expected

# --- Main ---
def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    args = parser.parse_args()
    print(args)
    
    # study = optuna.create_study(
    #    study_name="resnet34_parallel",
    #    direction="maximize",
    #    storage="sqlite:///optuna/storage/resnet34.db?timeout=600",
    #    load_if_exists=True
    # )

    study = optuna.load_study(
        study_name="resnet34_ndvi",
        storage="sqlite:///../optuna/storage/ndvi/resnet34.db",
    )
 
    study.optimize(objective, n_trials=1)

    print("\nBest trial:")
    print(study.best_trial.params)

    # Plotting & saving ONLY in the last job
    # To avoid conflicts, let only the highest trial ID handle visualization
    #if all_trials_completed(study, n_trials_expected=10):
    #    scores_path = "optuna/scores/resnet34/trial"
    #    plot_optuna_study(study, scores_path)
    #    joblib.dump(study, "optuna/studies/resnet34.pkl")

if __name__ == '__main__':
    main()
