import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_exploration import data_exploration
from split_data import preparation
from train_resnet50 import train_resnet
from evaluate_resnet50 import evaluate_resnet
from train_densenet import train_densenet
from evaluate_densenet import evaluate_densenet
from train_custom_model import train_custom
from evaluate_custom_model import evaluate_custom
from compare_models import compare_models

def run_script(script, model = None, version = None, models = None, versions = None):

    if script == 'explore':
        data_exploration()
    if script == 'preparation':
        preparation()
    if script == 'train' and model == 'resnet':
        train_resnet(version)
        evaluate_resnet(version)
    if script == 'train' and model == 'densenet':
        train_densenet(version)
        evaluate_densenet(version)
    if script == 'train' and model == 'custom':
        train_custom(version)
        evaluate_custom(version)
    if script == 'compare':
        compare_models(models, versions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run the data exploration, data splitting, model training and evaluation, and compare models.")

    parser.add_argument("--script", type = str, required = True, help = "REQUIRED. Options: explore, preparation, train, compare")
    parser.add_argument("--model", type = str, help = "REQUIRED FOR MODEL TRAINING: name of model for training and evaluation, either resnet, densenet, or custom.")
    parser.add_argument("--version", type = int, help = "version number, defaults to 1, expects and int.")
    parser.add_argument("--models", nargs = "+", type = str, help = "models to be compared: defaults to ['resnet', 'densenet', 'custom'], expects at least one input e.g., --models resnet densenet")    
    parser.add_argument("--versions", nargs = "+", type = int, help = "version numbers: defaults to [1, 1, 1], expects at least one input e.g., --versions 2 2.")
    
    args = parser.parse_args()

    run_script(args.script, args.model, args.version, args.models, args.versions)
