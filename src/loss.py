import torch
import torch.nn as nn
from utils import intersection_over_union_

class Loss(nn.module):
    def __init__(self, S=7, B=2, C=20):
        super(Loss, self).__init__()
        self.S = S
        self.B=B
        self.C=C
        """
        [QUOTE]
        we increase the loss from bounding box
        coordinate predictions and decrease the loss from confidence 
        predictions for boxes that don’t contain objects. We
        use two parameters, λcoord and λnoobj to accomplish this. We
        set λcoord = 5 and λnoobj = .5.
        """
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        # prediction[..., :] of shape (20 classes + box1(x, y, w, h, c) + box2(x, y, w, h, c))
        # target[..., :] os shape (20 classes + target_box(x, y, w, h, c)) because only one target box per cell
        iou_1 = intersection_over_union_(prediction[..., 21:25], target[..., 21:25])
        iou_2 = intersection_over_union_(prediction[..., 26:30], target[..., 26:30])
        
        """
        [QUOTE]
        YOLO predicts multiple bounding boxes per grid cell.
        At training time we only want one bounding box predictor
        to be responsible for each object. We assign one predictor
        to be “responsible” for predicting an object based on which
        prediction has the highest current IOU with the ground
        truth. 
        """

        # 1. choose box responsible for detecting the cell's object (highest IOU)