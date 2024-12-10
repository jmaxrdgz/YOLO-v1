import torch
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms

class PascalVOCDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, labels_path, csv_path, S=7, B=2, C=20, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.df = pd.read_csv(csv_path)
        self.transform = transform or transforms.ToTensor()
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index, ):
        label_path = os.path.join(self.labels_path, self.df.iloc[index, 1])
        image_path = os.path.join(self.images_path, self.df.iloc[index, 0])
        boxes = []

        with open(label_path, 'r') as file:
            for line in file:
                label_class, x, y, w, h = map(float, line.split())
                label_class = int(label_class)
                boxes.append([label_class, x, y, w, h])
        boxes = torch.tensor(boxes)

        image = Image.open(image_path)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        for box in boxes:
            i_cell, j_cell = int(self.S * box[1]), int(self.S * box[2])
            x_cell, y_cell = self.S * box[1] - i_cell, self.S * box[2] - j_cell
            w_cell, h_cell = box[3] * self.S, box[4] * self.S
            
            if label[i_cell, j_cell, 20] == 0:
                label[i_cell, j_cell, 20] = 1
                label[i_cell, j_cell, 21:25] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label[i_cell, j_cell, label_class] = 1

        return image, label