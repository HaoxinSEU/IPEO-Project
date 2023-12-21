import numpy as np


# count pixels for user accuracy 
def count_for_user_accuracy(pred, target, cls = 0):
    # pred: [batch_size, 1, H, W]
    # target: [batch_size, 1, H, W]
    pred = pred.view(-1)
    target = target.view(-1)

    # remove the prediction that has no GT
    pred[target==2] = 2

    # get all prediction and target that are equal to the class
    pred_inds = pred == cls
    target_inds = target == cls

    # count the number of pixels that are correctly predicted as the class
    correct_count = (pred_inds[target_inds]).long().sum().data.cpu().item()

    # count the number of predictions
    total_count = pred_inds.long().sum().data.cpu().item()
    
    return correct_count, total_count


# count pixels for producer accuracy 
def count_for_producer_accuracy(pred, target, cls = 0):
    pred = pred.view(-1)
    target = target.view(-1)

    # remove the GT that has no prediction
    target[pred==2] = 2

    # get all prediction and target that are equal to the class
    pred_inds = pred == cls
    target_inds = target == cls

    # count the number of pixels that are correctly recalled
    correct_count = (target_inds[pred_inds]).long().sum().data.cpu().item()

    # count the number of GT
    total_count = target_inds.long().sum().data.cpu().item()

    return correct_count, total_count


# count pixels for overall accuracy among all classes
def count_for_overall_accuracy(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    # remove the prediction that has no GT
    pred[target==2] = 2

    # get all prediction and target that are equal, without no-label
    pred_inds = pred == target
    pred_inds[target==2] = 0

    # get the number of correct predictions
    correct_count = pred_inds.long().sum().data.cpu().item()

    # get the number of all predictions, without no-label
    total_count = pred.shape[0] - (target==2).long().sum().data.cpu().item()

    return correct_count, total_count
