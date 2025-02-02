import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def compare_models(models = ['resnet', 'densenet', 'custom'], versions = [1, 1, 1]): 

    if 'resnet' in models:
        index = models.index('resnet')
        models[index] = 'resnet50'
    if 'custom' in models:
        index = models.index('custom')
        models[index] = 'custom_model'

    model_scores = {}

    for index, model in enumerate(models):

        with open(f'metrics/{model}_scores_v{versions[index]}.pkl', 'rb') as f:
            model_scores[model] = pickle.load(f)

    keys = []

    for key in model_scores:
        keys.append(model_scores[key].values())

    scores = pd.DataFrame(list(keys), columns = ['ROC-AUC', 'F1'])

    model_names = []

    if 'resnet50' in models:
        model_names.append('ResNet50')
    if 'densenet' in models:
        model_names.append('DenseNet162')
    if 'custom_model' in models:
        model_names.append('Custom Model')

    scores['Models'] = model_names

    plt.figure(figsize = (15, 5))

    x = np.arange(len(scores))
    bar_width = 0.35

    ROC_bars = plt.bar(x - bar_width / 2, scores['ROC-AUC'], color = 'blue', width = bar_width, label = 'ROC-AUC Score')
    F1_bars = plt.bar(x + bar_width / 2, scores['F1'], color = 'purple', width = bar_width, label = 'F1-Score')

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(x, scores['Models'])
    plt.ylabel('Scores')
    plt.title('ROC-AUC and F1-Scores')

    for bar in ROC_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha = 'center', va = 'bottom', fontsize = 10)

    for bar in F1_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha = 'center', va = 'bottom', fontsize = 10)

    plt.legend();

    plt.savefig(f'images/model_comparison_{versions}.png', bbox_inches = 'tight')

    print('Model comparison plot saved.')
    print('Script complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Compare model performance")

    parser.add_argument("--models", nargs = "+", type = str, help = "models to be compared: defaults to ['resnet', 'densenet', 'custom'], expects at least one input e.g., --models resnet densenet")    
    parser.add_argument("--versions", nargs = "+", type = int, help = "version numbers: defaults to [1, 1, 1], expects at least one input e.g., --versions 2 2.")
    args = parser.parse_args()
    compare_models(args.models, args.versions)

    
