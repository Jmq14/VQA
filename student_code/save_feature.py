import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
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
    feat = feat.reshape((feat.shape[0], feat.shape[1], -1))

    for i in range(feat.shape[0]):
        path = os.path.join(
            output_path, os.path.splitext(image_path[i])[0] + '_resnet_feature')
        np.save(path, feat[i])


def save_features(args):
    train_dataset = VqaDataset(image_dir=args.train_image_dir,
                               question_json_file_path=args.train_question_path,
                               annotation_json_file_path=args.train_annotation_path,
                               image_filename_pattern="COCO_train2014_{}.jpg",
                               is_training=True)
    val_dataset = VqaDataset(image_dir=args.test_image_dir,
                             question_json_file_path=args.test_question_path,
                             annotation_json_file_path=args.test_annotation_path,
                             image_filename_pattern="COCO_val2014_{}.jpg",
                             is_training=False)

    train_data = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=args.num_data_loader_workers)
    val_data = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=args.num_data_loader_workers)

    model = models.resnet152(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-2])

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    for batch_id, batch_data in enumerate(train_data):
        extract_features(model, batch_data, args.output_path)

    for batch_id, batch_data in enumerate(val_data):
        extract_features(model, batch_data, args.output_path)

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
    args = parser.parse_args()
    save_features(args)