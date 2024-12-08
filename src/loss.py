import torch
import torch.nn as nn
from utils import intersection_over_union_

class Loss(nn.Module):
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
        """
        [QUOTE]
        YOLO predicts multiple bounding boxes per grid cell.
        At training time we only want one bounding box predictor
        to be responsible for each object. We assign one predictor
        to be “responsible” for predicting an object based on which
        prediction has the highest current IOU with the ground
        truth. 
        """
        prediction = prediction.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        # prediction[..., :] of shape (20 classes + box1(c, x, y, w, h) + box2(c, x, y, w, h))
        # target only has one target box per cell
        iou_1 = intersection_over_union_(prediction[..., 21:25], target[..., 21:25])
        iou_2 = intersection_over_union_(prediction[..., 26:30], target[..., 21:25])
        _, best_box = torch.max(torch.stack([iou_1, iou_2], dim=0), dim=0)
        exist_box = target[..., 20].unsqueeze(3)

        #===============#
        #    BOX LOSS   #
        #===============#
        x_pred = prediction[..., (self.C + best_box * 5 + 1):(self.C + best_box * 5 + 2)]
        y_pred = prediction[..., (self.C + best_box * 5 + 2):(self.C + best_box * 5 + 3)]
        x_targ = target[..., 21:22]
        y_targ = target[..., 22:23]
        x_loss = self.loss(torch.flatten(exist_box * x_pred, end_dim=-2), torch.flatten(exist_box * x_targ, end_dim=-2))
        y_loss = self.loss(torch.flatten(exist_box * y_pred, end_dim=-2), torch.flatten(exist_box * y_targ, end_dim=-2))
        center_coord_loss = self.lambda_coord * (x_loss + y_loss)

        w_pred = prediction[..., (self.C + best_box * 5 + 3):(self.C + best_box * 5 + 4)]
        h_pred = prediction[..., (self.C + best_box * 5 + 4):(self.C + best_box * 5 + 5)]
        w_targ = target[..., 23:24]
        h_targ = target[..., 24:25]
        w_pred = torch.sqrt(torch.abs(w_pred) + 1e-6)
        h_pred = torch.sqrt(torch.abs(h_pred) + 1e-6)
        w_targ = torch.sqrt(w_targ)
        h_targ = torch.sqrt(h_targ)
        w_loss = self.loss(torch.flatten(exist_box * w_pred, end_dim=-2), torch.flatten(exist_box * w_targ, end_dim=-2))
        h_loss = self.loss(torch.flatten(exist_box * h_pred, end_dim=-2), torch.flatten(exist_box * h_targ, end_dim=-2))
        box_dim_loss = self.lambda_coord * (w_loss + h_loss)

        box_loss = box_dim_loss + center_coord_loss

        #================#
        #    PROB LOSS   #
        #================#
        prob_pred = prediction[..., (self.C + best_box * 5):(self.C + best_box * 5 + 1)]
        prob_targ = target[..., 20:21]
        obj_prob_loss = self.loss(torch.flatten(exist_box * prob_pred), torch.flatten(exist_box * prob_targ))

        total_prob_pred = prediction[..., 20:21] + prediction[..., 25:26]
        noobj_prob_loss = self.lambda_noobj * self.loss(torch.flatten((1 - exist_box) * total_prob_pred, start_dim=1), torch.flatten((1 - exist_box) * prob_targ, start_dim=1))
        #=================#
        #    CLASS LOSS   #
        #=================#
        C_pred = prediction[..., 0:self.C]
        C_targ = target[..., 0:self.C]
        class_loss = self.loss(torch.flatten(exist_box * C_pred, end_dim=-2), torch.flatten(exist_box * C_targ, end_dim=-2))
        
        #=================#
        #    TOTAL LOSS   #
        #=================#
        loss = box_loss + obj_prob_loss + noobj_prob_loss + class_loss

        return loss