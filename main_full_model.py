from argparse import Namespace
from src.args_project import parse_args

args: Namespace = parse_args('DeepAssetAllocationTraining')

if __name__ == '__main__':
    from src.training import train_model
    train_model(args)