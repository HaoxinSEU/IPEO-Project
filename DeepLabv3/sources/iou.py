import numpy as np


def iou(pred, target, cls = 0):
    # do calculation for each prediction in the batch
    # pred: [batch_size, 1, H, W]
    # target: [batch_size, 1, H, W]
    ious = []
    pred_more_ratios = []
    pred_less_ratios = []

    for i in range(pred.shape[0]):
        iou_ratio, pred_more_ratio, pred_less_ratio = iou_one_image(pred[i], target[i], cls)
        ious.append(iou_ratio)
        pred_more_ratios.append(pred_more_ratio)
        pred_less_ratios.append(pred_less_ratio)
        
    return np.array(ious), np.array(pred_more_ratios), np.array(pred_less_ratios)

    
def iou_one_image(pred, target, cls = 0):  
    pred = pred.view(-1)
    target = target.view(-1)
    
    # remove the prediction that has no GT
    pred[target==2] = 2

    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    ####
    pred_more = pred_inds.long().sum().data.cpu().item() - intersection
    pred_less = target_inds.long().sum().data.cpu().item() - intersection
    ####
    if union > 0:
        iou_ratio = float(intersection) / float(max(union, 1))
        ####
        pred_more_ratio = float(pred_more) / float(max(union, 1))
        pred_less_ratio = float(pred_less) / float(max(union, 1))
        ####
    else:
        iou_ratio = np.nan
        pred_more_ratio = np.nan
        pred_less_ratio = np.nan
    return iou_ratio, pred_more_ratio, pred_less_ratio
