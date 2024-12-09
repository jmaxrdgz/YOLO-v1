import torch

def intersection_over_union(pred_boxes, targ_boxes):
    """
    Compute intersection over union for (x, y, w, h) coordinates

    Parameters:
        pred_boxes (tensor): bounding boxes prediction (batch_size, 4)
        targ_boxes (tensor): bounding boxes target (batch_size, 4)

    Returns :
        (tensor): intersection over union (batch_size, 1)
    """
    # translate x, y, w, h coordinates to x1, x2, y1, y2
    box1_x1 = pred_boxes[..., 0:1] - pred_boxes[..., 2:3] / 2
    box1_x2 = pred_boxes[..., 0:1] + pred_boxes[..., 2:3] / 2
    box1_y1 = pred_boxes[..., 1:2] - pred_boxes[..., 3:4] / 2
    box1_y2 = pred_boxes[..., 1:2] + pred_boxes[..., 3:4] / 2
    box2_x1 = targ_boxes[..., 0:1] - targ_boxes[..., 2:3] / 2
    box2_x2 = targ_boxes[..., 0:1] + targ_boxes[..., 2:3] / 2
    box2_y1 = targ_boxes[..., 1:2] - targ_boxes[..., 3:4] / 2 
    box2_y2 = targ_boxes[..., 1:2] + targ_boxes[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    x2 = torch.min(box1_x2, box2_x2)
    y1 = torch.max(box1_y1, box2_y1)
    y2 = torch.min(box1_y2, box2_y2)

    # Also verifies that the boxes intersect (otherwise: intersection = 0)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))

    # 1e-6 added to avoid dividing by 0
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def convert_coordinates(x_cell, y_cell, x_relative, y_relative, w_cell, h_cell, S=7):
    x = (x_cell + x_relative) / S
    y = (y_cell + y_relative) / S
    w = w_cell / S
    h = h_cell / S
    return x, y, w, h
