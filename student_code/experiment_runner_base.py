from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter


class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self, is_last=False):
        total = 0.
        correct = 0.
        for batch_id, batch_data in enumerate(self._val_dataset_loader):
            if not is_last and batch_id >= 20:
                break
            images = batch_data['image']
            # questions = torch.max(batch_data['question_encoding'], 0)[0]
            questions = batch_data['question_encoding']
            # answers = torch.max(batch_data['answer_encoding'], 1)[1]
            answers = batch_data['answer_encoding']
            if self._cuda:
                images = images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
            predicted_answer = self._model(images, questions) 
            ground_truth_answer = answers
            total += ground_truth_answer.size(0)
            correct += (torch.max(predicted_answer, 1)[1] == ground_truth_answer).sum().item()
        return correct / total
            
    def train(self):
        writer = SummaryWriter()
        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                # ============
                # TODO: Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.
                images = batch_data['image']
                # questions = torch.max(batch_data['question_encoding'], 0)[0]
                questions = batch_data['question_encoding']
                answers = batch_data['answer_encoding']
                # answers = torch.max(batch_data['answer_encoding'], 1)[1]
                if self._cuda:
                    images = images.cuda()
                    questions = questions.cuda()
                    answers = answers.cuda()
                predicted_answer = self._model(images, questions) 
                # print(predicted_answer)
                # print(answers)
                # print(images.shape, questions.shape, predicted_answer.shape, answers.shape)
                ground_truth_answer = answers 
                # ============

                # Optimize the model according to the predictions
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    writer.add_scalar('train/loss', loss, current_step)

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    writer.add_scalar('val/accuracy', val_accuracy, current_step)

        self._model.eval()
        val_accuracy = self.validate(is_last=True)
        print("Final val accuracy {}".format(val_accuracy))
        writer.add_scalar('val/accuracy', val_accuracy, self._num_epochs*num_batches)
