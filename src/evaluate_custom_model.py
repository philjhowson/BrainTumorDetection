import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torchcam.methods import GradCAM
from custom_functions import custom_resize, CustomModel
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('data/test_indices.pkl', 'rb') as f:
    test_indices = pickle.load(f)

test_transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),
                                     transforms.Lambda(custom_resize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = [0.5], std = [0.5])])

test_dataset = Subset(datasets.ImageFolder(root = 'scans', transform = test_transform), test_indices)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomModel().to(device)

version = 1

checkpoint = torch.load(f'models/custom_model_tumor_checkpoint_v{version}.pth', weights_only = True)
model.load_state_dict(checkpoint)

with open(f'metrics/custom_model_tumor_performance_v{version}.pkl', 'rb') as f:
    history = pickle.load(f)

fig, ax = plt.subplots(2, 2, figsize = (15, 10))

training_items = ['loss', 'roc_auc', 'f1-weighted', 'grad_norm']
test_items = ['val_loss', 'val_roc_auc', 'val_f1-weighted']
titles = ['Loss', 'ROC-AUC', 'Weighted F1-Score', 'Gradient']

ticks = [1]
for tick in list(range(1, len(history['loss']) + 1)):
    if tick % 5 == 0:
        ticks.append(tick)

for index, axes in enumerate(ax.flat):

    if index > len(test_items) - 1:
        axes.plot(range(1, len(history[training_items[index]]) + 1), history[training_items[index]], label = 'Training')
        axes.set_xticks(ticks)
        axes.set_title(f"{titles[index]} by Epoch")
        axes.set_xlabel('Epoch')
        axes.set_ylabel(f"{titles[index]}")
        axes.legend()
        break

    axes.plot(range(1, len(history[training_items[index]]) + 1), history[training_items[index]], label = 'Training')
    axes.plot(range(1, len(history[test_items[index]]) + 1), history[test_items[index]], label = 'Validation', c = 'purple')
    axes.set_xticks(ticks)
    axes.set_title(f"{titles[index]} by Epoch")
    axes.set_xlabel('Epoch')
    axes.set_ylabel(f"{titles[index]}")
    axes.legend()

plt.savefig(f'images/custom_model_history_v{version}.png', bbox_inches = 'tight')

model.eval()

test_labels = []
test_preds = []

with torch.no_grad():
    for test_images, test_labels_batch in test_loader:
        test_images = test_images.to(device)
        test_labels_batch_criterion = test_labels_batch.to(device).unsqueeze(1).float()
        test_outputs = model(test_images)

        test_preds_prob = torch.sigmoid(test_outputs)
        test_predicted = (test_preds_prob > 0.5).int()
        test_labels.extend(test_labels_batch.cpu().numpy())
        test_preds.extend(test_predicted.cpu().numpy())

test_roc_score = roc_auc_score(test_labels, test_preds)
test_f1_score = f1_score(test_labels, test_preds)
report = classification_report(test_labels, test_preds, target_names = ['No Tumor', 'Tumor'])

print(f"The Test ROC-AUC Score: {round(test_roc_score, 3)}\n")
print(f"The Test F1 Score: {round(test_f1_score, 3)}\n")
print(report)

report_dict = classification_report(test_labels, test_preds, target_names = ['No Tumor', 'Tumor'], output_dict = True)

with open("metrics/custom_model_classification_report.pkl", "wb") as f:
    pickle.dump(report_dict, f)

scores = {'ROC-AUC' : test_roc_score,
          'F1 Score' : test_f1_score}

with open("metrics/custom_model_scores.pkl", "wb") as f:
    pickle.dump(scores, f)

grad_cam = GradCAM(model, model.block7[-3])

dataset = datasets.ImageFolder(root = 'scans')
no_set = dataset.samples[:98]
yes_set = dataset.samples[98:]
items = [48, 55,  2, 72, 53, 12, 64, 83, 82, 21,  3,  5, 79, 32, 59, 81, 33, 58,  7, 20]
activations = []
predicted_labels = []
labels = []
probability = []

for index, image in enumerate(items):

    if index < 10:
        img = Image.open(no_set[image][0]).convert('L')
        label = no_set[image][1]

    else:
        img = Image.open(yes_set[image][0]).convert('L')
        label = yes_set[image][1]
        
    input_tensor = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.enable_grad():
        out = model(input_tensor)  # Forward pass
    
    probabilities = torch.sigmoid(out)
    predicted_class = (probabilities > 0.5).long().item()
    predicted_prob = probabilities.item()
    activation_map = grad_cam(0, out)

    # Append results
    activations.append(activation_map)
    predicted_labels.append(predicted_class)
    probability.append(predicted_prob)
    labels.append(label)

grad_cam.remove_hooks()

mappings = {0: 'No Tumor', 1: 'Tumor'}

labels = [mappings.get(item, item) for item in labels]
predicted_labels = [mappings.get(item, item) for item in predicted_labels]

transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),
                                transforms.Lambda(custom_resize),
                                transforms.Lambda(lambda img: img.convert('RGB')),
                                transforms.ToTensor(),
                               ])

fig, ax = plt.subplots(4, 5, figsize = (20, 15))

no_set = dataset.samples[:98]
yes_set = dataset.samples[98:]

for index, axes in enumerate(ax.flat):
    if index < 10:
        with Image.open(no_set[items[index]][0]) as img:
            img = transform(img)
    else:
        with Image.open(yes_set[items[index]][0]) as img:
            img = transform(img)
    
    activation_map = activations[index]
    activation_map = activation_map[0].cpu().squeeze()
    
    activation_map = activation_map - activation_map.min()
    activation_map = activation_map / activation_map.max()
    activation_map = activation_map.unsqueeze(0).unsqueeze(0)
    
    # Resize the activation map using bilinear interpolation
    activation_map_resized = F.interpolate(activation_map, size = (400, 400), mode='bilinear', align_corners=False)
    
    heatmap = plt.cm.jet(activation_map_resized.squeeze().cpu().numpy())  # Generate heatmap
    heatmap = np.delete(heatmap, 3, axis=-1)  # Remove alpha channel
    
    # Normalize the heatmap
    heatmap = heatmap / (heatmap.max() + 1e-10)

    # Convert img to numpy and ensure it's in the correct shape (H, W, C)
    image = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)

    # Overlay the heatmap on the image
    overlayed_image = (image * (1 - 0.5) * 255 + heatmap * 0.5 * 255).astype(np.uint8)

    # Display the overlayed image
    axes.imshow(overlayed_image)
    if predicted_labels[index] == 'No Tumor':
        axes.set_title(f"Predicted: {predicted_labels[index]} ({round((1 - probability[index]) * 100, 2)}%),\nActual: {labels[index]}")
    else:
        axes.set_title(f"Predicted: {predicted_labels[index]} ({round(probability[index] * 100, 2)}%),\nActual: {labels[index]}")
    axes.axis('off')

plt.tight_layout();

plt.savefig(f'images/custom_models_grad_cam_v{version}.png', bbox_inches = 'tight')
