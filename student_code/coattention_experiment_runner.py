from student_code.coattention_net import CoattentionNet
from student_code.experiment_runner_base import ExperimentRunnerBase
from student_code.vqa_dataset import VqaDataset

import torch.optim as optim
import torch.nn as nn

class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        train_dataset = VqaDataset(image_dir=train_image_dir,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg")
        val_dataset = VqaDataset(image_dir=test_image_dir,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg")

        model = CoattentionNet(n_emb=512,
                               n_img=2048,
                               n_ques=len(train_dataset.dictionary),
                               n_ans=len(train_dataset.answers))

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs,
                         num_data_loader_workers=num_data_loader_workers)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=4e-4, alpha=0.99, eps=1e-8)

    def _optimize(self, predicted_answers, true_answer_ids):
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_answers, true_answer_ids)
        loss.backward()
        self.optimizer.step()

        return loss.item()
