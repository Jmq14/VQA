import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import os

from student_code.vqa_dataset import VqaDataset


def extract_features(model, batch_data, output_path):
    images = batch_data['image']
    if torch.cuda.is_available():
        images = images.cuda()

    image_path = batch_data['image_path']

    feat = model(images)
    feat = feat.data.cpu().numpy()
    # print(feat.shape)
    feat = feat.reshape((feat.shape[0], feat.shape[1], -1))
    # print(feat.shape)

    for i in range(feat.shape[0]):
        path = os.path.join(
            output_path, os.path.splitext(image_path[i])[0] + '_resnet_feature')
        # print(path)
        np.save(path, feat[i])


def save_features(args):

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_dataset = VqaDataset(image_dir=args.train_image_dir,
                               question_json_file_path=args.train_question_path,
                               annotation_json_file_path=args.train_annotation_path,
                               image_filename_pattern="COCO_train2014_{}.jpg",
                               is_training=True,
                               transform=transform)
    val_dataset = VqaDataset(image_dir=args.test_image_dir,
                             question_json_file_path=args.test_question_path,
                             annotation_json_file_path=args.test_annotation_path,
                             image_filename_pattern="COCO_val2014_{}.jpg",
                             is_training=False,
                             transform=transform)

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_data_loader_workers)
    val_data = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_data_loader_workers)

    model = models.resnet152(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    for batch_id, batch_data in enumerate(train_data):
        print('Training data {}/{}'.format(batch_id, len(train_data)))
        extract_features(model, batch_data, os.path.join(args.output_path, 'train2014'))

    for batch_id, batch_data in enumerate(val_data):
        print('Validation data {}/{}'.format(batch_id, len(train_data)))
        extract_features(model, batch_data, os.path.join(args.output_path, 'val2014'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load feature.')
    parser.add_argument('--train_image_dir', type=str)
    parser.add_argument('--train_question_path', type=str)
    parser.add_argument('--train_annotation_path', type=str)
    parser.add_argument('--test_image_dir', type=str)
    parser.add_argument('--test_question_path', type=str)
    parser.add_argument('--test_annotation_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--num_data_loader_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()
    save_features(args)
