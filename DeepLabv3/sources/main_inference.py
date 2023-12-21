import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import glob
from tqdm import tqdm
import argparse

# local import
import custom_model
from iou import iou
from accuracy import count_for_user_accuracy, count_for_producer_accuracy, count_for_overall_accuracy


# test dataset
class InferenceDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, folder_path, mode):
        super(InferenceDataset, self).__init__()
        # get all image filenames
        self.img_files = glob.glob(os.path.join(folder_path, 'input', mode, '*.*'))

        # get all targets (GT)
        self.label_files = []
        for img_path in self.img_files:
            image_filename, _ = os.path.splitext(os.path.basename(img_path))
            image_filename = image_filename.split('_', 1)  # get image ID
            label_filename_with_ext = f"target_{image_filename[1]}.tif"
            self.label_files.append(os.path.join(folder_path, 'target', mode, label_filename_with_ext))

        self.transforms = transforms.Compose([
            transforms.Resize(size=(512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __getitem__(self, index):
        img_path = self.img_files[index]
        label_path = self.label_files[index]

        image = Image.open(img_path)
        label = Image.open(label_path)

        # replace 255 and 244 by 2 in the label
        label_np = np.array(label)
        label_np[label_np > 1] = 2  

        # apply transforms
        image = self.transforms(image)

        return image, label_np

    def __len__(self):
        return len(self.img_files)


def main(data_dir, data_mode, weights_dir, num_classes):
    # detect if we have GPU or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # import our trained model
    model = custom_model.initialize_model(num_classes, keep_feature_extract=True)
    state_dict = torch.load(weights_dir, map_location=device)
    model = model.to(device)
    model.load_state_dict(state_dict)

    # set the model in evaluation mode
    model.eval()

    # load the test set
    test_dataset = InferenceDataset(data_dir, data_mode)
    # we fix the batch size to 1, because the images have different sizes
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    print("Starting to do inference...")

    running_iou = [[] for i in range(0, num_classes-1)]
    running_ua_correct_count = [0 for i in range(0, num_classes-1)]
    running_ua_total_count = [0 for i in range(0, num_classes-1)]
    running_pa_correct_count = [0 for i in range(0, num_classes-1)]
    running_pa_total_count = [0 for i in range(0, num_classes-1)]
    running_oa_correct_count = 0
    running_oa_total_count = 0

    # Iterate over test set
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # do the inference
        outputs = model(inputs)["out"]

        # select the prediction only in the first two classes
        outputs = outputs[:, :num_classes-1, :, :]
        _, preds = torch.max(outputs, 1)

        # convert the prediction from 512x512 to label's size
        preds = torch.nn.functional.interpolate(preds.unsqueeze(1).float(), size=(labels.size()[-2:]), mode="nearest").squeeze(1).long()

        # IoU calculation
        for i in range(0, num_classes-1):
            ious, _, _ = iou(preds, labels, i)
            running_iou[i].append(ious)

        # accuracy calculation for each class, without no-label
        for i in range(0, num_classes-1):
            ua_correct_count, ua_total_count = count_for_user_accuracy(preds, labels, i)
            pa_correct_count, pa_total_count = count_for_producer_accuracy(preds, labels, i)
            running_ua_correct_count[i] += ua_correct_count
            running_ua_total_count[i] += ua_total_count
            running_pa_correct_count[i] += pa_correct_count
            running_pa_total_count[i] += pa_total_count

        oa_correct_count, oa_total_count = count_for_overall_accuracy(preds, labels)
        running_oa_correct_count += oa_correct_count
        running_oa_total_count += oa_total_count

    # print the accuracy
    user_accuracy_0 = running_ua_correct_count[0] / running_ua_total_count[0]
    user_accuracy_1 = running_ua_correct_count[1] / running_ua_total_count[1]
    producer_accuracy_0 = running_pa_correct_count[0] / running_pa_total_count[0]
    producer_accuracy_1 = running_pa_correct_count[1] / running_pa_total_count[1]
    overall_accuracy = running_oa_correct_count / running_oa_total_count
    print('Inference Non-forest User Accuracy is: ', user_accuracy_0)
    print('Inference Forest User Accuracy is: ', user_accuracy_1)
    print('Inference Non-forest Producer Accuracy is: ', producer_accuracy_0)
    print('Inference Forest Producer Accuracy is: ', producer_accuracy_1)
    print('Inference Overall Accuracy is: ', overall_accuracy)

    # print IoU
    iou_0 = np.nanmean(np.array(running_iou[0]))
    iou_1 = np.nanmean(np.array(running_iou[1]))
    print('Inference Non-forest IoU is: ', iou_0)
    print('Inference Forest IoU is: ', iou_1)


# parse the arguments
def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", help='Specify the dataset directory path, should contain input/test and target/test')
    parser.add_argument(
        "weights_dir", help='Specify the  directory where model weights shall be stored.')
    parser.add_argument(
        "--data_mode", type=str, help="Specify the mode of the dataset (train, val or test)")
    parser.add_argument(
        "--num_classes", default=3, type=int, help="Number of classes in the dataset, no-label should be included in the count")
    
    args = parser.parse_args()

    main(args.data_dir, args.data_mode, args.weights_dir, args.num_classes)


if __name__ == "__main__":
    args_preprocess()