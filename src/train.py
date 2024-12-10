import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader


from model import YOLO_v1
from loss import Loss
from dataset import PascalVOCDataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

SEED = 42
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 IRL
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = ""
SAVE_MODEL = True
SAVE_MODEL_FILE = "checkpoint/overfit_100examples.pth.tar"
IMG_DIR = "/content/pascalvoc-yolo/images"
LABEL_DIR = "/content/pascalvoc-yolo/labels"

torch.manual_seed(SEED)

#================#
#   TRANSFORMS   #
#================#

class Compose_(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, bboxes):
        for t in self.transforms:
            image, bboxes = t(image), bboxes
        return image, bboxes
 
transform = Compose_([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

#====================#
#   TRAIN FUNCTION   #
#====================#

def train_epoch(train_loader, model, optimizer, loss_fn):
    mean_loss = []

    for batch, (x,y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'mean loss : {sum(mean_loss)/len(mean_loss)}')

#==============#
#   TRAINING   #
#==============#

def main():
    model = YOLO_v1().to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
    
    loss_fn = Loss()
    train_dataset = PascalVOCDataset(
        images_path=IMG_DIR, 
        labels_path=LABEL_DIR, 
        csv_path='/content/pascalvoc-yolo/100examples.csv', 
        transform=transform
        )
    test_dataset = PascalVOCDataset(
        images_path=IMG_DIR, 
        labels_path=LABEL_DIR, 
        csv_path='/content/pascalvoc-yolo/test.csv', 
        transform=transform # verif si besoin de transform
        )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=True
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    for epoch in range(EPOCHS):
        prediction_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mAP = mean_average_precision(
            prediction_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mAP}")
        train_epoch(train_loader, model, optimizer, loss_fn)

    if SAVE_MODEL:
        checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
        save_checkpoint(checkpoint, filename=SAVE_MODEL_FILE)

if __name__=="__main__":
    main()