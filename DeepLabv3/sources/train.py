import os
import torch
import numpy as np
import time
import copy
from tqdm import tqdm

from iou import iou


def train_model(model, num_classes, dataloaders, criterion, optimizer, scheduler, device, dest_dir, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    counter = 0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_ious = {}
            running_mores = {}
            running_lesss = {}

            for i in range(0, num_classes-1):
                running_ious[i] = []
                running_mores[i] = []
                running_lesss[i] = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Security, skip this iteration if the batch_size is 1
                if 1 == inputs.shape[0]:
                    print("Skipping iteration because batch_size = 1")
                    continue

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # calculate IoU, pred_more, pred_less for each class
                for cls in range(0, num_classes-1):
                    ious, pred_more_ratio, pred_less_ratio = iou(preds, labels, cls)
                    running_ious[cls].append(ious)
                    running_mores[cls].append(pred_more_ratio)
                    running_lesss[cls].append(pred_less_ratio)

                # Increment counter
                counter = counter + 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # update the learning rate
            if phase == 'val':
                scheduler.step(epoch_loss)

            # calculate mean IoU, pred_more, pred_less
            running_iou_means = {}

            print('#############################################')
            print("running_ious[0] shape", np.concatenate(running_ious[0]).shape)
            print("running_ious[1] shape", np.concatenate(running_ious[1]).shape)
            print("running_mores[1] shape", np.concatenate(running_mores[1]).shape)
            print("running_lesss[1] shape", np.concatenate(running_lesss[1]).shape)
            
            running_iou_means[0] = np.nanmean(np.concatenate(running_ious[0]))
            running_iou_means[1] = np.nanmean(np.concatenate(running_ious[1]))
            running_more_means = np.nanmean(np.concatenate(running_mores[1]))
            running_less_means = np.nanmean(np.concatenate(running_lesss[1]))

            epoch_acc = (running_iou_means[0] + running_iou_means[1]) / 2
            print('{} Loss: {:.4f} Non-forest IoU: {:.4f} Forest IoU: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, running_iou_means[0], running_iou_means[1], epoch_acc))
            print('More prediction on forest:  {:.4f}, less prediction on forest:  {:.4f}'.format(running_more_means, running_less_means))
            print('#############################################')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # Save current model every 20 epochs
            if 0 == epoch % 20:
                current_model_path = os.path.join(dest_dir, f"checkpoint_{epoch:04}_DeepLabV3_weight.pth")
                print(f"Save current model : {current_model_path}")
                torch.save(model.state_dict(), current_model_path)

                current_best_model_path = os.path.join(dest_dir, f"checkpoint_{epoch:04}_DeepLabV3_weight_best.pth")
                print(f"Save current best model : {current_best_model_path}")
                torch.save(best_model_state_dict, current_best_model_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_model_state_dict, val_acc_history
