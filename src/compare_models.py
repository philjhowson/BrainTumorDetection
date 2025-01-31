import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

with open('metrics/resnet_50_scores.pkl', 'rb') as f:
    resnet_scores = pickle.load(f)

with open('metrics/densenet_scores.pkl', 'rb') as f:
    densenet_scores = pickle.load(f)

with open('metrics/custom_model_scores.pkl', 'rb') as f:
    custom_model_scores = pickle.load(f)

scores = pd.DataFrame(list([resnet_scores.values(), densenet_scores.values(),
                            custom_model_scores.values()]), columns = ['ROC-AUC',
                                                                       'F1'])

scores['Models'] = ['Resnet 50', 'Densenet 169', 'Custom Model']

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

plt.savefig('images/model_comparison.png', bbox_inches = 'tight')
