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
import argparse

# --- Dataset ---
class SatelliteWealthDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        index_paths = row['image_paths']  # Dict: {'ndvi': path1, 'vari': path2, ...}

        channels = []
        for path in index_paths.values():
            with rasterio.open(path) as src:
                img = src.read(1).astype(np.float32)
                img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                img = (img - img.mean()) / (img.std() + 1e-7)  # Normalize
                channels.append(torch.tensor(img))

        img_tensor = torch.stack(channels, dim=0)  # Shape: (C, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = torch.tensor(row['wealthpooled'], dtype=torch.float32)
        return img_tensor, label

def match_multichannel_images_to_labels(csv_path, image_root_folder, indices, target_column):
    dhs_df = pd.read_csv(csv_path)
    image_groups = {}

    for idx_name in indices:
        folder = os.path.join(image_root_folder, idx_name.upper(), f"{idx_name}_images")
        for fname in os.listdir(folder):
            if fname.endswith('.tif'):
                base = fname.replace('.tif', '')
                _, lat, lon, year = base.split('_')
                key = f"{lat}_{lon}_{year}"
                path = os.path.join(folder, fname)
                image_groups.setdefault(key, {})[idx_name] = path

    matches = []
    for key, paths in image_groups.items():
        if len(paths) != len(indices):
            continue  # Skip if any index is missing

        lat, lon, year = map(float, key.split('_'))
        distances = np.sqrt((dhs_df['LATNUM'] - lat)**2 + (dhs_df['LONGNUM'] - lon)**2)
        nearest_idx = distances.idxmin()
        wealth_value = dhs_df.loc[nearest_idx, target_column]
        matches.append({'image_paths': paths, 'wealthpooled': wealth_value})

    return pd.DataFrame(matches)

# --- Training function ---
def train_model(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, num_epochs, patience=10, use_cuda=False):
    train_losses = []
    test_r2s = []
    best_r2 = -float('inf')
    epochs_no_improve = 0

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
            for imgs, labels in train_loader:
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

        if r2 > best_r2 + 1e-4:  # Small delta to avoid stopping on noise
            best_r2 = r2
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- Early stopping condition ---
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no R² improvement for {patience} epochs.")
            break

    return train_losses, test_r2s, max(test_r2s)

def build_efficientnet_b3(input_channels=6, dropout_rate=0.3):
    base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    base.features[0][0] = nn.Conv2d(input_channels, 40, kernel_size=3, stride=2, padding=1, bias=False)

    class EfficientNetRegressor(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(p=dropout_rate)
            self.regressor = nn.Linear(1536, 1)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.dropout(x)
            x = self.regressor(x)
            return x

    return EfficientNetRegressor(base)
    
# --- Optuna objective function ---
def objective(trial, indices, model_name, input_channels=1):
    # Paths
    CSV_FILE = os.environ.get('WS_DIR') + '/dhs_wealth_index_cleaned.csv'
    ROOT_IMAGE_FOLDER = os.environ.get('WS_DIR') + f'/processed'
    TARGET_COLUMN = 'wealthpooled'

    num_epochs = 70
    patience = 20

    img_size = 224
    if model_name == "efficientnet":
        img_size = 300  # B3 default

    # Transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(degrees=30),
        T.Resize((img_size, img_size)),
    ])
    test_transform = T.Compose([T.Resize((img_size, img_size))])

    # Prepare dataset
    dataset_df = match_multichannel_images_to_labels(CSV_FILE, ROOT_IMAGE_FOLDER, indices, TARGET_COLUMN)
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42)
    train_dataset = SatelliteWealthDataset(train_df, transform=train_transform)
    test_dataset = SatelliteWealthDataset(test_df, transform=test_transform)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = None
    batch_size = None
    lr = None
    weight_decay = None
    dropout_rate = None
    optimizer_choice = None
    if model_name == "efficientnet":
        print(f"Model: EfficientNet-B3")

        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-5, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.3)
        optimizer_choice = trial.suggest_categorical("optimizer", ["Adam"])

        model = build_efficientnet_b3(input_channels, dropout_rate=dropout_rate)

    elif model_name == "vgg":
        print(f"Model: VGG-16")

        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.3, 0.5)
        optimizer_choice = trial.suggest_categorical("optimizer", ["Adam"])

        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.classifier[-1].in_features, 1)
        )
        
    else:
        print(f"Model: ResNet-34")

        batch_size = trial.suggest_categorical("batch_size", [64, 128])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.4)
        optimizer_choice = trial.suggest_categorical("optimizer", ["Adam"])

        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(model.fc.in_features, 1)
        )

    index_name = "_".join(indices)

    print(f"Index: {index_name}")
    print(f"Batch Size: {batch_size} - LR: {lr} - Weight Decay: {weight_decay} - Dropout Rate: {dropout_rate}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=use_cuda)
    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",      # because higher R² is better
        factor=0.5,      # shrink LR by 2x
        patience=5,      # wait 5 epochs with no R² improvement
        cooldown=1,      # brief cooldown after a step
        min_lr=1e-6,     # don’t vanish LR
        verbose=True
    )
    
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Train model
    train_losses, test_r2s, best_r2 = train_model(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, num_epochs=num_epochs, patience=patience, use_cuda=use_cuda)

    TRIAL_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", trial.number))  # Fallback if running locally

    model_output_dir = f"../optuna/models/multi/{model_name}/{index_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    # Save best model for this trial
    torch.save(model.state_dict(), f"{model_output_dir}/trial_{TRIAL_ID}.pth")

    r2_scores_output_dir = f"../optuna/r2_scores/multi/{model_name}/{index_name}"
    os.makedirs(r2_scores_output_dir, exist_ok=True)
    # Save R² scores per epoch for this trial
    np.save(f"{r2_scores_output_dir}/trial_{TRIAL_ID}.npy", np.array(test_r2s))

    return best_r2

def create_objective(indices, model_name, input_channels):
    def objective_wrapper(trial):
        return objective(trial, indices, model_name, input_channels)
    return objective_wrapper

# --- Main ---
def main():
    allowed_indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]
    allowed_models = ["resnet34", "efficientnet", "vgg"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--indices", nargs='+', required=True, choices=allowed_indices,
                    help=f"Indices must be a subset of: {', '.join(allowed_indices)}")
    # parser.add_argument("--index", type=str, required=True, choices=allowed_indices,
    #                help=f"Index must be one of: {', '.join(allowed_indices)}")
    parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                    help=f"Model must be one of: {', '.join(allowed_models)}")

    args = parser.parse_args()
    input_channels = len(args.indices)

    objective_wrapper = create_objective(args.indices, args.model, input_channels)

    index_name = "_".join(args.indices)

    study = optuna.load_study(
        study_name=f'{args.model}_{index_name}',
        storage=f"sqlite:///../optuna/storage/multi/{args.model}/{args.model}_{index_name}.db",
    )
 
    study.optimize(objective_wrapper, n_trials=1)

    print("\nBest trial:")
    print(study.best_trial.params)

if __name__ == '__main__':
    main()
