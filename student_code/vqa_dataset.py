from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import glob
import os
import skimage.io as io

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.image_idx_list = self.vqa.getImgIds()
        self.ques_idx_list = self.vqa.getQuesIds()
        self.image_dir = image_dir
        self.image_filename_pattern = image_filename_pattern
        self.dictionary = self._build_dictionary()

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, idx):
        ann = self.vqa.loadQA(idx)[0]
        image_id = ann['image_id']
        image_path = os.path.join(self.image_dir, self.image_filename_pattern.format(image_id))
        image = io.imread(image_path)

    def _build_dictionary(self):
        pass
