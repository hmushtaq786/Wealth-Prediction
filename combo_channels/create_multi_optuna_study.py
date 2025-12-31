import optuna
import argparse
import os

allowed_indices = ["ndvi", "vari", "msavi", "mndwi", "ndbi", "ndmi"]
allowed_models = ["resnet", "efficientnet", "vgg"]

parser = argparse.ArgumentParser()
parser.add_argument("--indices", nargs='+', required=True, choices=allowed_indices,
                    help=f"Indices must be a subset of: {', '.join(allowed_indices)}")
parser.add_argument("--model", type=str, required=True, choices=allowed_models,
                    help=f"Model must be one of: {', '.join(allowed_models)}")
args = parser.parse_args()

storage_dir = f"../optuna/storage/multi/{args.model}"
os.makedirs(storage_dir, exist_ok=True) 

index_name = "_".join(args.indices)

optuna.create_study(
    study_name=f'{args.model}_{index_name}', 
    storage=f'sqlite:///{storage_dir}/{args.model}_{index_name}.db', 
    direction='maximize'
    )