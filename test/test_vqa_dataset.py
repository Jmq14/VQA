import unittest
import os
from torch.utils.data import DataLoader
from student_code.vqa_dataset import VqaDataset


class TestVqaDataset(unittest.TestCase):

    def test_load_dataset(self):
        """
        This method gives you a quick way to run your dataset to make sure it loads files correctly.
        It doesn't assert a particular result from indexing the dataset; that will depend on your design.
        Feel free to fill in more asserts here, to validate your design.
        """
        # Arrange
        current_dir = os.path.dirname(__file__)
        question_file = os.path.join(current_dir, "test_questions.json")
        annotation_file = os.path.join(current_dir, "test_annotations.json")

        vqa_dataset = VqaDataset(question_json_file_path=question_file,
                                 annotation_json_file_path=annotation_file,
                                 image_dir=current_dir,
                                 image_filename_pattern="COCO_train2014_{}.jpg")

        # Act
        vqa_len = len(vqa_dataset)
        dataset_item = vqa_dataset[0]

        # Assert
        self.assertEqual(vqa_len, 2)
        self.assertTrue(type(dataset_item) is dict)

    def test_use_dataset_loader(self):
        """
        Verify that the dataset can be successfully loaded using the DatasetLoader class.
        """
        # Arrange
        current_dir = os.path.dirname(__file__)
        question_file = os.path.join(current_dir, "test_questions.json")
        annotation_file = os.path.join(current_dir, "test_annotations.json")

        vqa_dataset = VqaDataset(question_json_file_path=question_file,
                                 annotation_json_file_path=annotation_file,
                                 image_dir=current_dir,
                                 image_filename_pattern="COCO_train2014_{}.jpg")
        dataset_loader = DataLoader(vqa_dataset, batch_size=2)

        # Act & Assert - the test will fail if iterating through the data loader fails
        for id, data in enumerate(dataset_loader):
            # Not doing anything here. Feel free to fill this in, if you like.
            pass
