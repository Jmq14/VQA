from student_code.simple_baseline_net import SimpleBaselineNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch.nn as nn
import torch.optim as optim


class SimpleBaselineExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Simple Baseline model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   is_training=True)
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 is_training=False)

        model = SimpleBaselineNet(len(train_dataset.dictionary), len(train_dataset.answers))
        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD([
            {'params': model.ques_feat.parameters(), 'lr': 0.8},
            {'params': model.fc.parameters()}], lr=0.01, momentum=0.9)

    def _optimize(self, predicted_answers, true_answer_ids):
        loss = self.criterion(predicted_answers, true_answer_ids)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data.item() 
        
