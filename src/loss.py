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
        # target only has one target box per cell (important !)
        iou_1 = intersection_over_union_(prediction[..., 21:25], target[..., 21:25])
        iou_2 = intersection_over_union_(prediction[..., 26:30], target[..., 21:25])
        _, best_box_i = torch.max(torch.stack([iou_1, iou_2], dim=0), dim=0) # (BATCH_SIZE, S, S, 1)
        exist_box = target[..., 20].unsqueeze(3) # (BATCH_SIZE, S, S, 1)

        #===============#
        #    BOX LOSS   #
        #===============#
        best_box_coord = (1 - best_box_i) * prediction[..., 21:25] + best_box_i * prediction[..., 26:30] # (BATCH_SIZE, S, S, 4)
        target_box_coord = target[..., 21:25] # (BATCH_SIZE, S, S, 4)
        box_center_loss = (
            self.loss(
                torch.flatten(exist_box * best_box_coord[..., 0:1], end_dim=-2),
                torch.flatten(exist_box * target_box_coord[..., 0:1], end_dim=-2)
            ) 
            +
            self.loss(
                torch.flatten(exist_box * best_box_coord[..., 1:2], end_dim=-2),
                torch.flatten(exist_box * target_box_coord[..., 1:2], end_dim=-2)
            )
        )

        box_dim_loss = (
            self.loss(
                torch.flatten(exist_box * 
                              torch.sign(best_box_coord[..., 2:3]) * 
                              torch.sqrt(torch.abs(best_box_coord[..., 2:3]) + 1e-6)),
                torch.flatten(exist_box * torch.sqrt(target_box_coord[..., 2:3] + 1e-6))
            )
            +
            self.loss(
                torch.flatten(exist_box * 
                              torch.sign(best_box_coord[..., 2:3]) *
                              torch.sqrt(torch.abs(best_box_coord[..., 3:4]) + 1e-6)),
                torch.flatten(exist_box * torch.sqrt(target_box_coord[..., 3:4] + 1e-6))
            )
        )

        box_loss = self.lambda_coord * (box_center_loss + box_dim_loss) # Scalar

        #================#
        #    PROB LOSS   #
        #================#
        best_box_prob = (1 - best_box_i) * prediction[..., 20:21] + best_box_i * prediction[..., 25:26]
        target_prob = target[..., 20:21]
        obj_prob_loss = (
            self.loss(
                torch.flatten(exist_box * best_box_prob, start_dim=1), 
                torch.flatten(exist_box * target_prob, start_dim=1)
                )
        )

        noobj_prob_loss = (
            self.loss(
                torch.flatten((1 - exist_box) * prediction[..., 20:21], start_dim=1), 
                torch.flatten((1 - exist_box) * target_prob, start_dim=1)
                )
            +
            self.loss(
                torch.flatten((1 - exist_box) * prediction[..., 25:26], start_dim=1), 
                torch.flatten((1 - exist_box) * target_prob, start_dim=1)
            )
        )
        
        prob_loss = obj_prob_loss + self.lambda_noobj * noobj_prob_loss

        #=================#
        #    CLASS LOSS   #
        #=================#
        class_pred = prediction[..., 0:20]
        class_targ = target[..., 0:20]
        class_loss = (
            self.loss(
                torch.flatten(exist_box * class_pred, end_dim=-2), 
                torch.flatten(exist_box * class_targ, end_dim=-2))
        )
        
        #=================#
        #    TOTAL LOSS   #
        #=================#
        loss = box_loss + prob_loss + class_loss # Not normalized by batch size in the original paper

        return loss
    
def test():
    S, B, C = 7, 2, 20  # Grid size, number of boxes, number of classes
    batch_size = 4
    prediction = torch.rand((batch_size, S * S * (C + B * 5)))
    target = torch.rand((batch_size, S, S, C + 5))
    loss_fn = Loss(S=S, B=B, C=C)
    loss = loss_fn.forward(prediction, target)
    print(loss)