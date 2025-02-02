import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from custom_functions import custom_resize, EarlyStopping, CustomModel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pickle

def train_custom(version = 1):

    train_transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),
                                    transforms.Lambda(custom_resize),
                                    transforms.RandomVerticalFlip(p = 0.5),
                                    transforms.RandomHorizontalFlip(p = 0.5),
                                    transforms.RandomAffine(degrees = (-30, 30),
                                                            translate = (0.1, 0.1),
                                                            fill = 0),
                                    transforms.GaussianBlur(kernel_size = 5, sigma = (0.1, 2.0)),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(scale = (0.02, 0.33), ratio = (0.3, 3.3), value = 0),
                                    transforms.Normalize(mean = [0.5], std = [0.5])])

    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels = 1),
                                         transforms.Lambda(custom_resize),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean = [0.5], std = [0.5])])

    with open('data/train_indices.pkl', 'rb') as f:
        train_indices = pickle.load(f)

    with open('data/val_indices.pkl', 'rb') as f:
        val_indices = pickle.load( f)

    train_dataset = Subset(datasets.ImageFolder(root = 'scans', transform = train_transform), train_indices)
    val_dataset = Subset(datasets.ImageFolder(root = 'scans', transform = test_transform), val_indices)

    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 16, shuffle = False)

    """
    Calculates class weights for the loss function.
    """
    labels = [label for _, label in train_dataset]
    sklearn_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(labels), y = labels)
    minority, majority = torch.tensor(sklearn_weights).float().numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomModel().to(device)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode = 'fan_out', nonlinearity = 'relu')
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3)
    early_stopping = EarlyStopping(patience = 10, reload = 'ROC_AUC')

    epochs = 100
    history = {'loss': [],  'f1-weighted' : [], 'roc_auc': [], 'grad_norm': [], 'val_loss': [], 'val_roc_auc': [],  'val_f1-weighted' : []}

    print(f"Custom model loaded. Ready to train version {version}")

    for epoch in range(epochs):
        
        model.train()
        
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            hot_labels = labels.unsqueeze(1).float()

            weights = torch.tensor([minority if label == 1 else majority for label in labels]).to(device).unsqueeze(1)
            criterion = torch.nn.BCEWithLogitsLoss(weight = weights)
            
            loss = criterion(outputs, hot_labels)
            loss.backward()
            
            """
            calculate gradient norms for monitoring, crucial to ensure that we don't get vanishing or exploding
            gradients. Calculates other metrics (as above) for observation of model performance over time. 
            """
            batch_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    batch_grad_norm += param.grad.norm(2).item() ** 2
            batch_grad_norm = batch_grad_norm ** 0.5

            optimizer.step()

            running_loss += loss.item()
            
            preds_prob = torch.sigmoid(outputs)
            predicted = (preds_prob > 0.5).int()    
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_roc_score = roc_auc_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average = 'weighted')

        history['loss'].append(epoch_loss)
        history['roc_auc'].append(epoch_roc_score)
        history['f1-weighted'].append(epoch_f1)
        history['grad_norm'].append(batch_grad_norm)

        """
        after completion of an epoch, performs model and evaluation of the model performance on a validation (test)
        batch. This is done by doing an epoch from the test loader, torch.no_grad() is used in this case because
        calculating gradients is not necessary for evaluation.
        """
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for val_images, val_labels_batch in val_loader:
                val_images, val_labels_batch = val_images.to(device), val_labels_batch
                val_labels_batch_criterion = val_labels_batch.to(device).unsqueeze(1).float()
                val_outputs = model(val_images)

                weights = torch.tensor([minority if label == 1 else majority for label in val_labels_batch]).to(device).unsqueeze(1)
                criterion = torch.nn.BCEWithLogitsLoss(weight = weights)
                
                val_loss += criterion(val_outputs, val_labels_batch_criterion).item()

                val_preds_prob = torch.sigmoid(val_outputs)
                val_predicted = (val_preds_prob > 0.5).int()
                val_labels.extend(val_labels_batch.cpu().numpy())
                val_preds.extend(val_predicted.cpu().numpy())

        val_loss /= len(val_loader)
        val_roc_score = roc_auc_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average = 'weighted')

        history['val_loss'].append(val_loss)
        history['val_f1-weighted'].append(val_f1)
        history['val_roc_auc'].append(val_roc_score)
        
        """
        prints the metrics.
        """
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Loss: {epoch_loss:.4f}, Gradient: {batch_grad_norm:.4f}, "
              f"ROC-AUC Score: {epoch_roc_score:.4f}, F1-Score: {epoch_f1:.4f} "
              f"Val Loss: {val_loss:.4f}, Val ROC-AUC Score: {val_roc_score:.4f}, Val F1-Score: {val_f1:.4f}")

        if early_stopping(val_loss, val_f1, val_roc_score, model):

            break
            
        """
        step the scheduler so that learning rate can be adjusted (if necessary).
        """
        scheduler.step(val_loss)

    """
    after all epoches are performed the best ROC model is reloaded, and the model,
    and the history dictionary are saved so that I can continue training later if desired.
    """
    model.load_state_dict(early_stopping.best_ROC_model)
    torch.save(model.state_dict(), f'models/custom_model_tumor_checkpoint_v{version}.pth')

    with open(f'metrics/custom_model_tumor_performance_v{version}.pkl', 'wb') as f:
        pickle.dump(history, f)

    print('Custom model and training metrics saved.')
    print('Script complete')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train the custom model.")
    parser.add_argument("--version", type = int, help = "version number, defaults to 1, expects and int.")
    args = parser.parse_args()
    train_custom(args.version)
