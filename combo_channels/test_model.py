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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import optuna
import sys
import joblib
import argparse
import scipy.stats
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Try to import albumentations, fall back to basic transforms if not available
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("Albumentations not available, using basic transforms")
    ALBUMENTATIONS_AVAILABLE = False

# --- Enhanced Dataset ---
class ImprovedSatelliteWealthDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_spatial_features=True, quality_filter=True):
        if quality_filter:
            self.dataframe = self.filter_low_quality_samples(dataframe)
        else:
            self.dataframe = dataframe
        self.transform = transform
        self.use_spatial_features = use_spatial_features

    def __len__(self):
        return len(self.dataframe)

    def filter_low_quality_samples(self, df, quality_threshold=0.1):
        """Remove samples with poor image quality"""
        filtered_samples = []
        
        for idx, row in df.iterrows():
            paths = row['image_paths']
            quality_scores = []
            
            try:
                for path in paths.values():
                    with rasterio.open(path) as src:
                        img = src.read(1).astype(np.float32)
                        # Quality metric: ratio of valid pixels
                        valid_ratio = np.sum(~np.isnan(img)) / img.size
                        quality_scores.append(valid_ratio)
                
                if min(quality_scores) > quality_threshold:
                    filtered_samples.append(row)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
                
        return pd.DataFrame(filtered_samples).reset_index(drop=True)

    def robust_normalize(self, img, percentile_clip=2):
        """Robust normalization with outlier clipping"""
        # Handle invalid values
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip outliers
        if np.sum(img != 0) > 100:  # Ensure enough valid pixels
            low, high = np.percentile(img[img != 0], [percentile_clip, 100-percentile_clip])
            img = np.clip(img, low, high)
        
        # Robust normalization using median and MAD
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        if mad > 1e-7:
            normalized = (img - median) / mad
        else:
            normalized = img - median
            
        return normalized

    def extract_spatial_features(self, img):
        """Extract spatial statistics that correlate with wealth"""
        features = []
        
        # Basic statistics
        features.append(np.mean(img))
        features.append(np.std(img))
        features.append(scipy.stats.skew(img.flatten()))
        features.append(scipy.stats.kurtosis(img.flatten()))
        
        # Texture features
        try:
            # Gradient-based texture
            grad_x = np.gradient(img, axis=1)
            grad_y = np.gradient(img, axis=0)
            features.append(np.std(grad_x))
            features.append(np.std(grad_y))
            
            # Local Binary Pattern approximation
            features.append(np.std(ndimage.generic_filter(img, np.std, size=3)))
            
        except:
            # Fallback values if texture computation fails
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        index_paths = row['image_paths']

        channels = []
        spatial_features_list = []
        
        for path in index_paths.values():
            try:
                with rasterio.open(path) as src:
                    img = src.read(1).astype(np.float32)
                    img = self.robust_normalize(img)
                    
                    if self.use_spatial_features:
                        spatial_features = self.extract_spatial_features(img)
                        spatial_features_list.append(spatial_features)
                    
                    channels.append(torch.tensor(img))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create dummy channel if loading fails
                dummy_img = np.zeros((224, 224), dtype=np.float32)
                channels.append(torch.tensor(dummy_img))
                if self.use_spatial_features:
                    spatial_features_list.append(np.zeros(7, dtype=np.float32))

        img_tensor = torch.stack(channels, dim=0)  # Shape: (C, H, W)

        if self.transform:
            if ALBUMENTATIONS_AVAILABLE:
                # Convert to numpy for albumentations
                img_np = img_tensor.numpy().transpose(1, 2, 0)
                transformed = self.transform(image=img_np)
                img_tensor = transformed['image']
            else:
                # Apply torchvision transforms channel by channel
                transformed_channels = []
                for i in range(img_tensor.shape[0]):
                    channel = img_tensor[i]
                    # Convert single channel to 3-channel for PIL compatibility
                    channel_3d = channel.unsqueeze(0).repeat(3, 1, 1)
                    transformed_channel = self.transform(channel_3d)[0]  # Take only first channel back
                    transformed_channels.append(transformed_channel)
                img_tensor = torch.stack(transformed_channels, dim=0)

        label = torch.tensor(row['wealthpooled'], dtype=torch.float32)
        
        if self.use_spatial_features and spatial_features_list:
            spatial_features = torch.tensor(np.concatenate(spatial_features_list), dtype=torch.float32)
            return img_tensor, spatial_features, label
        else:
            return img_tensor, label

# --- Enhanced Data Matching ---
def match_multichannel_images_to_labels(csv_path, image_root_folder, indices, target_column, max_distance=0.01):
    """Improved matching with distance threshold"""
    dhs_df = pd.read_csv(csv_path)
    image_groups = {}

    for idx_name in indices:
        folder = os.path.join(image_root_folder, idx_name.upper(), f"{idx_name}_images")
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist")
            continue
            
        for fname in os.listdir(folder):
            if fname.endswith('.tif'):
                try:
                    base = fname.replace('.tif', '')
                    parts = base.split('_')
                    if len(parts) >= 4:
                        _, lat, lon, year = parts[:4]
                        key = f"{lat}_{lon}_{year}"
                        path = os.path.join(folder, fname)
                        image_groups.setdefault(key, {})[idx_name] = path
                except Exception as e:
                    print(f"Error processing filename {fname}: {e}")
                    continue

    matches = []
    for key, paths in image_groups.items():
        if len(paths) != len(indices):
            continue  # Skip if any index is missing

        try:
            lat, lon, year = map(float, key.split('_'))
            distances = np.sqrt((dhs_df['LATNUM'] - lat)**2 + (dhs_df['LONGNUM'] - lon)**2)
            min_distance = distances.min()
            
            if min_distance <= max_distance:  # Only include close matches
                nearest_idx = distances.idxmin()
                wealth_value = dhs_df.loc[nearest_idx, target_column]
                if not np.isnan(wealth_value):  # Ensure valid wealth value
                    matches.append({
                        'image_paths': paths, 
                        'wealthpooled': wealth_value,
                        'distance': min_distance
                    })
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            continue

    print(f"Found {len(matches)} valid matches out of {len(image_groups)} image groups")
    return pd.DataFrame(matches)

# --- Enhanced Loss Functions ---
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        residual = torch.abs(pred - target)
        condition = residual < self.delta
        squared_loss = 0.5 * residual ** 2
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        return torch.where(condition, squared_loss, linear_loss).mean()

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_factor=2.0):
        super().__init__()
        self.weight_factor = weight_factor
    
    def forward(self, pred, target):
        # Give more weight to extreme poverty/wealth cases
        weights = 1 + self.weight_factor * torch.abs(target)
        mse = (pred - target) ** 2
        return (weights * mse).mean()

# --- Improved Model Architectures ---
class MultiScaleEfficientNet(nn.Module):
    def __init__(self, input_channels=6, dropout_rate=0.3, use_spatial_features=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B3
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        base.features[0][0] = nn.Conv2d(input_channels, 40, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.features = base.features
        self.use_spatial_features = use_spatial_features
        
        # Multi-scale pooling
        self.pool_scales = [1, 2, 4]
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.pool_scales
        ])
        
        # Calculate feature dimensions
        base_features = 1536
        multi_scale_features = sum(base_features // (scale**2) for scale in self.pool_scales)
        
        # Spatial features processing
        if use_spatial_features:
            spatial_feature_dim = len(['ndvi', 'vari', 'msavi', 'mndwi', 'ndmi', 'ndbi']) * 7  # 7 features per channel
            self.spatial_processor = nn.Sequential(
                nn.Linear(spatial_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2),
                nn.Linear(256, 128)
            )
            total_features = multi_scale_features + 128
        else:
            total_features = multi_scale_features
        
        # Enhanced regressor
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate/2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, spatial_features=None):
        # Extract CNN features
        x = self.features(x)
        
        # Multi-scale pooling
        pooled_features = []
        for pool in self.pools:
            pooled = pool(x).flatten(1)
            pooled_features.append(pooled)
        
        cnn_features = torch.cat(pooled_features, dim=1)
        
        # Combine with spatial features
        if self.use_spatial_features and spatial_features is not None:
            spatial_processed = self.spatial_processor(spatial_features)
            combined_features = torch.cat([cnn_features, spatial_processed], dim=1)
        else:
            combined_features = cnn_features
        
        return self.regressor(combined_features)

class ImprovedResNet(nn.Module):
    def __init__(self, input_channels=6, dropout_rate=0.3, use_spatial_features=True):
        super().__init__()
        
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Use ResNet50 instead of 34
        base.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.use_spatial_features = use_spatial_features
        
        base_features = 2048  # ResNet50 output
        
        if use_spatial_features:
            spatial_feature_dim = input_channels * 7
            self.spatial_processor = nn.Sequential(
                nn.Linear(spatial_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate/2),
                nn.Linear(256, 128)
            )
            total_features = base_features + 128
        else:
            total_features = base_features
        
        self.regressor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout_rate/2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, spatial_features=None):
        cnn_features = self.features(x).flatten(1)
        
        if self.use_spatial_features and spatial_features is not None:
            spatial_processed = self.spatial_processor(spatial_features)
            combined_features = torch.cat([cnn_features, spatial_processed], dim=1)
        else:
            combined_features = cnn_features
        
        return self.regressor(combined_features)

# --- Enhanced Data Augmentation ---
def get_train_transforms():
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
            A.Resize(224, 224),
            A.Normalize(mean=[0.0] * 6, std=[1.0] * 6),
            ToTensorV2()
        ])
    else:
        # Fallback to basic torchvision transforms
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

def create_simple_transforms(input_channels=6, image_size=224):
    """Create basic transforms that work regardless of albumentations availability"""
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=30),
        T.Resize((image_size, image_size)),
    ])
    
    test_transform = T.Compose([
        T.Resize((image_size, image_size))
    ])
    
    return train_transform, test_transform

# --- Improved Training Function ---
def train_model_improved(model, train_loader, test_loader, device, num_epochs=100, 
                        patience=15, use_spatial_features=True, use_cuda=False):
    
    # Initialize optimizer with different learning rates for different parts
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'features' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': 1e-4}       # Higher LR for new layers
    ], weight_decay=1e-5)
    
    # Enhanced loss function
    criterion = HuberLoss(delta=0.5)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    
    train_losses = []
    test_r2s = []
    test_maes = []
    best_r2 = -float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            if use_spatial_features:
                imgs, spatial_features, labels = batch_data
                imgs, spatial_features, labels = imgs.to(device), spatial_features.to(device), labels.to(device)
            else:
                imgs, labels = batch_data
                imgs, labels = imgs.to(device), labels.to(device)
                spatial_features = None

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda):
                if use_spatial_features:
                    preds = model(imgs, spatial_features).squeeze()
                else:
                    preds = model(imgs).squeeze()
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if use_spatial_features:
                    imgs, spatial_features, labels = batch_data
                    imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                    outputs = model(imgs, spatial_features).squeeze().cpu().numpy()
                else:
                    imgs, labels = batch_data
                    imgs = imgs.to(device)
                    outputs = model(imgs).squeeze().cpu().numpy()
                
                preds.extend(outputs)
                trues.extend(labels.numpy())

        r2 = r2_score(trues, preds)
        mae = mean_absolute_error(trues, preds)
        epoch_loss = running_loss / num_batches
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - R²: {r2:.4f} - MAE: {mae:.4f}")

        train_losses.append(epoch_loss)
        test_r2s.append(r2)
        test_maes.append(mae)

        if r2 > best_r2 + 1e-4:
            best_r2 = r2
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no R² improvement for {patience} epochs.")
            break

    return train_losses, test_r2s, test_maes, max(test_r2s)

# --- Cross-Validation ---
def cross_validate_model(dataset_df, model_builder, indices, k=5, use_spatial_features=True):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    cv_maes = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_df)):
        print(f"\n--- Fold {fold+1}/{k} ---")
        
        train_df = dataset_df.iloc[train_idx].reset_index(drop=True)
        val_df = dataset_df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        if ALBUMENTATIONS_AVAILABLE:
            train_transform = get_train_transforms()
            test_transform = get_test_transforms()
        else:
            train_transform, test_transform = create_simple_transforms(len(indices))
            
        train_dataset = ImprovedSatelliteWealthDataset(
            train_df, transform=train_transform, use_spatial_features=use_spatial_features
        )
        val_dataset = ImprovedSatelliteWealthDataset(
            val_df, transform=test_transform, use_spatial_features=use_spatial_features
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=use_cuda)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=use_cuda)
        
        # Build and train model
        model = model_builder(len(indices), use_spatial_features=use_spatial_features)
        model = model.to(device)
        
        _, _, _, best_r2 = train_model_improved(
            model, train_loader, val_loader, device, 
            num_epochs=50, patience=10, use_spatial_features=use_spatial_features, use_cuda=use_cuda
        )
        
        # Final evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if use_spatial_features:
                    imgs, spatial_features, labels = batch_data
                    imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                    outputs = model(imgs, spatial_features).squeeze().cpu().numpy()
                else:
                    imgs, labels = batch_data
                    imgs = imgs.to(device)
                    outputs = model(imgs).squeeze().cpu().numpy()
                
                preds.extend(outputs)
                trues.extend(labels.numpy())
        
        fold_r2 = r2_score(trues, preds)
        fold_mae = mean_absolute_error(trues, preds)
        
        cv_scores.append(fold_r2)
        cv_maes.append(fold_mae)
        
        print(f"Fold {fold+1} - R²: {fold_r2:.4f}, MAE: {fold_mae:.4f}")
    
    return np.mean(cv_scores), np.std(cv_scores), np.mean(cv_maes), np.std(cv_maes)

# --- Ensemble Methods ---
class ModelEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, test_loader, device, use_spatial_features=True):
        all_predictions = []
        
        for model in self.models:
            model.eval()
            preds = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    if use_spatial_features:
                        imgs, spatial_features, labels = batch_data
                        imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                        outputs = model(imgs, spatial_features).squeeze().cpu().numpy()
                    else:
                        imgs, labels = batch_data
                        imgs = imgs.to(device)
                        outputs = model(imgs).squeeze().cpu().numpy()
                    
                    preds.extend(outputs)
            
            all_predictions.append(preds)
        
        # Average predictions
        ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred

# --- Enhanced Objective Function ---
def enhanced_objective(trial, indices, model_name, input_channels=1, use_spatial_features=True):
    # Paths
    CSV_FILE = os.environ.get('WS_DIR', '.') + '/dhs_wealth_index_cleaned.csv'
    ROOT_IMAGE_FOLDER = os.environ.get('WS_DIR', '.') + f'/processed'
    TARGET_COLUMN = 'wealthpooled'

    num_epochs = 80
    patience = 20

    # Prepare dataset
    dataset_df = match_multichannel_images_to_labels(CSV_FILE, ROOT_IMAGE_FOLDER, indices, TARGET_COLUMN)
    
    if len(dataset_df) < 100:
        print(f"Warning: Only {len(dataset_df)} samples available. Consider relaxing matching criteria.")
        return 0.0
    
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42, stratify=None)
    
    # Create datasets with enhanced preprocessing
    if ALBUMENTATIONS_AVAILABLE:
        train_transform = get_train_transforms()
        test_transform = get_test_transforms()
    else:
        train_transform, test_transform = create_simple_transforms(input_channels)
    
    train_dataset = ImprovedSatelliteWealthDataset(
        train_df, transform=train_transform, use_spatial_features=use_spatial_features
    )
    test_dataset = ImprovedSatelliteWealthDataset(
        test_df, transform=test_transform, use_spatial_features=use_spatial_features
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Enhanced hyperparameter search
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
    
    # Loss function selection
    loss_type = trial.suggest_categorical("loss_type", ["huber", "mse", "weighted_mse"])
    
    if loss_type == "huber":
        delta = trial.suggest_float("huber_delta", 0.1, 2.0)
        criterion = HuberLoss(delta=delta)
    elif loss_type == "weighted_mse":
        weight_factor = trial.suggest_float("weight_factor", 1.0, 3.0)
        criterion = WeightedMSELoss(weight_factor=weight_factor)
    else:
        criterion = nn.MSELoss()

    # Model selection
    if model_name == "efficientnet":
        model = MultiScaleEfficientNet(
            input_channels, dropout_rate=dropout_rate, use_spatial_features=use_spatial_features
        )
    else:  # resnet or others
        model = ImprovedResNet(
            input_channels, dropout_rate=dropout_rate, use_spatial_features=use_spatial_features
        )

    model = model.to(device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=use_cuda)

    # Enhanced optimizer
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'features' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)

    # Enhanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    # Train model
    _, _, _, best_r2 = train_model_improved(
        model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, 
        num_epochs=num_epochs, patience=patience, use_spatial_features=use_spatial_features, use_cuda=use_cuda
    )

    # Save model and results
    TRIAL_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", trial.number))
    index_name = "_".join(indices)
    
    model_output_dir = f"../optuna/models/improved/{model_name}/{index_name}"
    os.makedirs(model_output_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_output_dir}/trial_{TRIAL_ID}.pth")

    return best_r2

def create_enhanced_objective(indices, model_name, input_channels, use_spatial_features=True):
    def objective_wrapper(trial):
        return enhanced_objective(trial, indices, model_name, input_channels, use_spatial_features)
    return objective_wrapper

# --- Main Function ---
def main():
    allowed_indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]
    allowed_models = ["resnet", "efficientnet"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--indices", nargs='+', required=True, choices=allowed_indices,
                       help=f"Indices must be a subset of: {', '.join(allowed_indices)}")
    parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                       help=f"Model must be one of: {', '.join(allowed_models)}")
    parser.add_argument("--use-spatial-features", action='store_true', default=True,
                       help="Use spatial features in addition to CNN features")
    parser.add_argument("--cross-validate", action='store_true', default=False,
                       help="Run cross-validation instead of single train/test split")

    args = parser.parse_args()
    input_channels = len(args.indices)

    if args.cross_validate:
        # Run cross-validation
        CSV_FILE = os.environ.get('WS_DIR', '.') + '/dhs_wealth_index_cleaned.csv'
        ROOT_IMAGE_FOLDER = os.environ.get('WS_DIR', '.') + f'/processed'
        TARGET_COLUMN = 'wealthpooled'
        
        dataset_df = match_multichannel_images_to_labels(CSV_FILE, ROOT_IMAGE_FOLDER, args.indices, TARGET_COLUMN)
        
        def model_builder(input_channels, use_spatial_features=True):
            if args.model == "efficientnet":
                return MultiScaleEfficientNet(input_channels, use_spatial_features=use_spatial_features)
            else:
                return ImprovedResNet(input_channels, use_spatial_features=use_spatial_features)
        
        mean_r2, std_r2, mean_mae, std_mae = cross_validate_model(
            dataset_df, model_builder, args.indices, k=5, use_spatial_features=args.use_spatial_features
        )
        
        print(f"\nCross-Validation Results:")
        print(f"R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        
    else:
        # Run Optuna optimization
        objective_wrapper = create_enhanced_objective(args.indices, args.model, input_channels, args.use_spatial_features)
        
        index_name = "_".join(args.indices)
        
        # Create study directory
        study_dir = f"../optuna/storage/improved/{args.model}"
        os.makedirs(study_dir, exist_ok=True)
        
        study_name = f'{args.model}_{index_name}_improved'
        storage_url = f"sqlite:///{study_dir}/{args.model}_{index_name}_improved.db"
        
        # Try to load existing study, create new one if it doesn't exist
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url,
            )
            print(f"Loaded existing study: {study_name}")
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction='maximize',  # We want to maximize R² score
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            print(f"Created new study: {study_name}")
     
        study.optimize(objective_wrapper, n_trials=1)

        print("\nBest trial:")
        print(study.best_trial.params)
        print(f"Best R² Score: {study.best_trial.value:.4f}")

# --- Additional Utility Functions ---
def analyze_channel_importance(model, test_loader, device, use_spatial_features=True):
    """Analyze which channels contribute most to predictions"""
    model.eval()
    
    baseline_preds = []
    channel_importance = {}
    
    with torch.no_grad():
        for batch_data in test_loader:
            if use_spatial_features:
                imgs, spatial_features, labels = batch_data
                imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                
                # Baseline prediction
                baseline_pred = model(imgs, spatial_features)
                baseline_preds.extend(baseline_pred.cpu().numpy())
                
                # Test each channel ablation
                for ch in range(imgs.shape[1]):
                    masked_imgs = imgs.clone()
                    masked_imgs[:, ch, :, :] = 0  # Zero out channel
                    
                    masked_pred = model(masked_imgs, spatial_features)
                    importance = torch.abs(baseline_pred - masked_pred).mean().item()
                    
                    if ch not in channel_importance:
                        channel_importance[ch] = []
                    channel_importance[ch].append(importance)
            else:
                imgs, labels = batch_data
                imgs = imgs.to(device)
                
                baseline_pred = model(imgs)
                baseline_preds.extend(baseline_pred.cpu().numpy())
                
                for ch in range(imgs.shape[1]):
                    masked_imgs = imgs.clone()
                    masked_imgs[:, ch, :, :] = 0
                    
                    masked_pred = model(masked_imgs)
                    importance = torch.abs(baseline_pred - masked_pred).mean().item()
                    
                    if ch not in channel_importance:
                        channel_importance[ch] = []
                    channel_importance[ch].append(importance)
    
    # Average importance scores
    avg_importance = {ch: np.mean(scores) for ch, scores in channel_importance.items()}
    return avg_importance

def create_prediction_report(model, test_loader, device, true_labels, use_spatial_features=True):
    """Create comprehensive prediction report"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if use_spatial_features:
                imgs, spatial_features, labels = batch_data
                imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                outputs = model(imgs, spatial_features).squeeze().cpu().numpy()
            else:
                imgs, labels = batch_data
                imgs = imgs.to(device)
                outputs = model(imgs).squeeze().cpu().numpy()
            
            predictions.extend(outputs)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    r2 = r2_score(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    mse = mean_squared_error(true_labels, predictions)
    rmse = np.sqrt(mse)
    
    # Correlation
    correlation = np.corrcoef(true_labels, predictions)[0, 1]
    
    report = {
        'r2_score': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'predictions': predictions,
        'true_values': true_labels
    }
    
    return report

def plot_predictions_vs_actual(predictions, true_values, save_path=None):
    """Create scatter plot of predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(true_values, predictions, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(min(true_values), min(predictions))
    max_val = max(max(true_values), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Wealth Index')
    plt.ylabel('Predicted Wealth Index')
    plt.title('Predictions vs Actual Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2 = r2_score(true_values, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_model_with_metadata(model, optimizer, scheduler, epoch, r2_score, save_path, metadata=None):
    """Save model with comprehensive metadata"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'r2_score': r2_score,
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_class': model.__class__.__name__,
        'metadata': metadata or {}
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path} with R² score: {r2_score:.4f}")

def load_model_with_metadata(model, optimizer, scheduler, load_path):
    """Load model with metadata"""
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded from {load_path}")
    print(f"Epoch: {checkpoint['epoch']}, R² Score: {checkpoint['r2_score']:.4f}")
    print(f"Saved at: {checkpoint['timestamp']}")
    
    return checkpoint['epoch'], checkpoint['r2_score'], checkpoint.get('metadata', {})

# --- Training with Enhanced Logging ---
def train_model_improved(model, train_loader, test_loader, device, optimizer, criterion, scheduler, scaler, 
                        num_epochs=100, patience=15, use_spatial_features=True, use_cuda=False, log_dir=None):
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    train_losses = []
    test_r2s = []
    test_maes = []
    learning_rates = []
    best_r2 = -float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            if use_spatial_features and len(batch_data) == 3:
                imgs, spatial_features, labels = batch_data
                imgs, spatial_features, labels = imgs.to(device), spatial_features.to(device), labels.to(device)
            else:
                imgs, labels = batch_data
                imgs, labels = imgs.to(device), labels.to(device)
                spatial_features = None

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda):
                if use_spatial_features and spatial_features is not None:
                    preds = model(imgs, spatial_features).squeeze()
                else:
                    preds = model(imgs).squeeze()
                loss = criterion(preds, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1

        # Evaluate
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                if use_spatial_features and len(batch_data) == 3:
                    imgs, spatial_features, labels = batch_data
                    imgs, spatial_features = imgs.to(device), spatial_features.to(device)
                    outputs = model(imgs, spatial_features).squeeze().cpu().numpy()
                else:
                    imgs, labels = batch_data
                    imgs = imgs.to(device)
                    outputs = model(imgs).squeeze().cpu().numpy()
                
                preds.extend(outputs)
                trues.extend(labels.numpy())

        r2 = r2_score(trues, preds)
        mae = mean_absolute_error(trues, preds)
        epoch_loss = running_loss / num_batches
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - R²: {r2:.4f} - MAE: {mae:.4f} - LR: {current_lr:.2e}")

        train_losses.append(epoch_loss)
        test_r2s.append(r2)
        test_maes.append(mae)

        if r2 > best_r2 + 1e-4:
            best_r2 = r2
            epochs_no_improve = 0
            
            # Save best model
            if log_dir:
                best_model_path = os.path.join(log_dir, 'best_model.pth')
                save_model_with_metadata(model, optimizer, scheduler, epoch, r2, best_model_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no R² improvement for {patience} epochs.")
            break
    
    # Save training history
    if log_dir:
        history = {
            'train_losses': train_losses,
            'test_r2s': test_r2s,
            'test_maes': test_maes,
            'learning_rates': learning_rates
        }
        np.save(os.path.join(log_dir, 'training_history.npy'), history)

    return train_losses, test_r2s, test_maes, max(test_r2s)

if __name__ == '__main__':
    main()