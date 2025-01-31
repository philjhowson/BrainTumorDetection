from torchvision import datasets
from sklearn.model_selection import train_test_split
import pickle

dataset = datasets.ImageFolder(root = 'scans')

targets = dataset.targets
train_indices, temp_indices = train_test_split(list(range(len(targets))), stratify = targets, test_size = 0.2, random_state = 42)
val_indices, test_indices = train_test_split(list(range(len(targets))), stratify = targets, test_size = 0.5, random_state = 42)

with open('data/train_indices.pkl', 'wb') as f:
    pickle.dump(train_indices, f)

with open('data/val_indices.pkl', 'wb') as f:
    pickle.dump(val_indices, f)

with open('data/test_indices.pkl', 'wb') as f:
    pickle.dump(test_indices, f)
