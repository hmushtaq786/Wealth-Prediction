import optuna

optuna.create_study(
    study_name='resnet34', 
    storage='sqlite:///optuna/storage/resnet34.db', 
    direction='maximize'
    )

optuna.create_study(
    study_name='efficientnetb3', 
    storage='sqlite:///optuna/storage/efficientnetb3.db', 
    direction='maximize'
    )

optuna.create_study(
    study_name='vgg16', 
    storage='sqlite:///optuna/storage/vgg16.db', 
    direction='maximize'
    )