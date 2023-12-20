import os
import torch
import numpy as np
import time
import copy
from tqdm import tqdm

# local import
from iou import iou


def train_model(model, num_classes, dataloaders, criterion, optimizer, scheduler, device, dest_dir, num_epochs=25):
    since = time.time()

    val_IoU_history = []

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_IoU = 0.0

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

            for i in range(0, num_classes-1):
                running_ious[i] = []

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
                    ious, _, _ = iou(preds, labels, cls)
                    running_ious[cls].append(ious)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # update the learning rate
            if phase == 'val':
                scheduler.step(epoch_loss)

            # calculate mean IoU, pred_more, pred_less
            running_iou_means = {}

            print('#############################################')
            running_iou_means[0] = np.nanmean(np.concatenate(running_ious[0]))
            running_iou_means[1] = np.nanmean(np.concatenate(running_ious[1]))

            mIoU = (running_iou_means[0] + running_iou_means[1]) / 2
            print('{} Loss: {:.4f} Non-forest IoU: {:.4f} Forest IoU: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, running_iou_means[0], running_iou_means[1], mIoU))
            print('#############################################')

            # deep copy the model
            if phase == 'val' and running_iou_means[1] > best_IoU:
                best_IoU = running_iou_means[1]
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_IoU_history.append(mIoU)

            # Save current model every 20 epochs
            if 0 == epoch % 20:
                current_model_path = os.path.join(dest_dir, f"checkpoint_{epoch:04}_DeepLabV3_weight.pth")
                print(f"Save current model : {current_model_path}")
                torch.save(model.state_dict(), current_model_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_IoU))

    # load best model weights
    return best_model_state_dict, val_IoU_history
