import optuna
import argparse
import os

allowed_models = ["resnet", "efficientnet", "vgg"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                    help=f"Model must be one of: {', '.join(allowed_models)}")
args = parser.parse_args()

storage_dir = f"../optuna/storage/raw_images/{args.model}"
os.makedirs(storage_dir, exist_ok=True) 

optuna.create_study(
    study_name=f'{args.model}_images', 
    storage=f'sqlite:///{storage_dir}/{args.model}_images.db', 
    direction='maximize'
    )