import numpy as np
from PIL import Image
import torch
from torch import nn

def custom_resize(img):
    
    img_array = np.array(img)
    locations = np.where(img_array >= 15)
    min_row, max_row = np.min(locations[0]), np.max(locations[0])
    min_col, max_col = np.min(locations[1]), np.max(locations[1])
    trimmed_image = img_array[min_row : max_row + 1, min_col : max_col + 1]

    img = Image.fromarray(trimmed_image)

    aspect_ratio = img.width / img.height
    
    if img.width > img.height:
            new_width = 400
            new_height = int(new_width / aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            if img.height < 400:
                value = 400 - img.height
                half2 = value // 2
                half1 = value - half2
                top = np.zeros((half1, 400), dtype = np.uint8)
                bottom = np.zeros((half2, 400), dtype = np.uint8)
                img = np.array(img)
                img = np.concatenate((top, img, bottom), axis = 0)
                img = Image.fromarray(img)
        
    else:
        new_height = 400
        new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
            
        if img.width < 400:
            value = 400 - img.width
            half2 = value // 2
            half1 = value - half2
            left = np.zeros((400, half1), dtype = np.uint8)
            right = np.zeros((400, half2), dtype = np.uint8)
            img = np.array(img)
            img = np.concatenate((left, img, right), axis = 1)
            img = Image.fromarray(img)

    return img

class EarlyStopping:
    def __init__(self, patience = 5, reload = 'ROC_AUC'):
        self.patience = patience
        self.best_loss = float('inf')
        self.best_f1 = 0
        self.best_ROC = 0
        self.counter = 0
        self.best_f1_model = None
        self.best_ROC_model = None
        self.reload = reload

    def __call__(self, val_loss, f1_score, ROC_AUC, model):
        if f1_score > self.best_f1:
            self.best_f1_model = model.state_dict()
        if ROC_AUC > self.best_ROC:
            self.best_ROC_model = model.state_dict()
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter = 0
                print("Early stopping triggered")
                if self.reload == 'ROC_AUC':
                    model.load_state_dict(self.best_ROC_model)
                else:
                    model.load_state_dict(self.best_f1_model)
                        
                return True
                    
        return False

class CustomResNet50(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet50, self).__init__()

        self.base_model = base_model
        
        self.skip_connection1 = nn.Conv2d(2048, 2048, kernel_size = 1, stride = 1, padding = 0)

        self.Conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

        self.skip_connection2 = nn.Conv2d(2048, 1024, kernel_size = 1, stride = 1, padding = 0)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

        self.skip_connection3 = nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0)

        self.Conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

        self.AAPool2d = nn.AdaptiveAvgPool2d((1, 1))
        
        self.Flatten = nn.Flatten(start_dim = 1)

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1))
    
    def forward(self, x):

        x = self.base_model(x)

        skip_connection1 = self.skip_connection1(x)
        x = self.Conv1(x)
        x = x + skip_connection1
        
        skip_connection2 = self.skip_connection2(x)
        x = self.Conv2(x)
        x = x + skip_connection2
        
        skip_connection3 = self.skip_connection3(x)
        x = self.Conv3(x)
        x = x + skip_connection3

        x = self.AAPool2d(x)
        x = self.Flatten(x)
        x = self.fc(x)

        return x

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.large = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 7, stride = 1, padding = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.medium = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size = 5, stride = 2, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.residual1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.intermediate1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU()
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.intermediate2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.intermediate3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )
        self.residual4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.intermediate4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU()
        )
        self.residual5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU()
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.intermediate6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )
        self.residual7 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size = 1, stride = 1, padding = 0),
            nn.ReLU()
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.AAPool2d = nn.AdaptiveAvgPool2d((1, 1))
        
        self.Flatten = nn.Flatten(start_dim = 1)

        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1))

    def forward(self, x):

        large = self.large(x)
        medium = self.medium(x)
        x = torch.cat([large, medium], axis = 1)
        
        x = self.block1(x)

        residual = self.residual1(x)
        x = self.block2(x)
        x = x + residual

        intermediate = self.intermediate1(residual)
        residual = self.residual2(x)
        intermediate = torch.cat([intermediate, residual], axis = 1)
        x = self.block3(x)
        x = x + intermediate

        intermediate = self.intermediate2(intermediate)
        residual = self.residual3(x)
        intermediate = torch.cat([intermediate, residual], axis = 1)
        x = self.block4(x)
        x = x + intermediate

        intermediate = self.intermediate3(intermediate)
        residual = self.residual4(x)
        intermediate = torch.cat([intermediate, residual], axis = 1)
        x = self.block5(x)
        x = x + intermediate

        intermediate = self.intermediate4(intermediate)
        residual = self.residual5(x)
        intermediate = torch.cat([intermediate, residual], axis = 1)
        x = self.block6(x)
        x = x + intermediate
        
        x = self.block7(x)
        
        x = self.AAPool2d(x)
        x = self.Flatten(x)

        x = self.fc(x)

        return x
