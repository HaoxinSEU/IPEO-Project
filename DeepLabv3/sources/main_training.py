import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import pathlib

# Local import
from custom_dataset import DatasetSegmentation
from custom_model import initialize_model
from train import train_model


def main(data_dir, dest_dir, num_classes, batch_size, num_epochs, keep_feature_extract, train_forest_balance, val_remove_no_forest, weight):

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: DatasetSegmentation(data_dir, x) for x in ['train', 'val']}
    # Balance the training dataset
    if train_forest_balance:
        image_datasets['train'].balance_minority_class()
    
    # Remove samples from the validation dataset that have no forest
    if val_remove_no_forest:
        image_datasets['val'].remove_no_forest_val()
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    print("Initializing Model...")

    # Initialize model
    model_deeplabv3 = initialize_model(num_classes, keep_feature_extract)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model_deeplabv3 = model_deeplabv3.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    # finetuning we will be updating all parameters. However, if we are
    # doing feature extract method, we will only update the parameters
    # that we have just initialized

    params_to_update = model_deeplabv3.parameters()
    print("Params to learn:")
    if keep_feature_extract:
        params_to_update = []
        for name, param in model_deeplabv3.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_deeplabv3.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-4)
    
    # setup the scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=True)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss(weight=(torch.FloatTensor(weight).to(device) if weight else None))

    # Prepare output directory
    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)

    print("Train...")

    # Train and evaluate
    model_deeplabv3_state_dict, hist = train_model(model_deeplabv3, num_classes, dataloaders_dict, criterion, optimizer_ft, scheduler, device, dest_dir, num_epochs=num_epochs)

    print("Save ...")
    torch.save(model_deeplabv3_state_dict, os.path.join(dest_dir, "best_DeepLabV3_weights.pth"))


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help='Specify the dataset directory path, should contain input/train, target/train, input/val and target/val')
    parser.add_argument(
        "dest_dir", help='Specify the  directory where model weights shall be stored.')
    parser.add_argument("--num_classes", default=3, type=int, help="Number of classes in the dataset, no-label should be included in the count")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs to train for")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training (change depending on how much memory you have)")
    parser.add_argument("--keep_feature_extract", action="store_true", help="Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params")
    parser.add_argument("--train_forest_balance", action="store_true", help="Flag for balancing the training dataset. When True, we balance the training dataset")
    parser.add_argument("--val_remove_no_forest", action="store_true", help="Flag for removing samples from the validation dataset that have no forest. When True, we remove samples from the validation dataset that have no forest")
    parser.add_argument('-w', action='append', type=float, help="Add more weight to some classes. If this argument is used, then it should be called as many times as there are classes (see --num_classes)")

    args = parser.parse_args()

    # Build weight list
    weight = []
    if args.w:
        for w in args.w:
            weight.append(w)

    main(args.data_dir, args.dest_dir, args.num_classes, args.batch_size, args.epochs, args.keep_feature_extract, args.train_forest_balance, args.val_remove_no_forest, weight)


if __name__ == '__main__':
    args_preprocess()