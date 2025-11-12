import optuna
import argparse
import os

allowed_indices = ["ndvi", "vari", "msavi", "mndwi", "ndmi", "ndbi"]
allowed_models = ["resnet34", "efficientnet", "vgg"]

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str, required=True, choices=allowed_indices,
                    help=f"Index must be one of: {', '.join(allowed_indices)}")
parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                    help=f"Model must be one of: {', '.join(allowed_models)}")
args = parser.parse_args()

storage_dir = f"../optuna/storage/{args.model}"
os.makedirs(storage_dir, exist_ok=True) 

optuna.create_study(
    study_name=f'{args.model}_{args.index}', 
    storage=f'sqlite:///{storage_dir}/{args.model}_{args.index}.db', 
    direction='maximize'
    )