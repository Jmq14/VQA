from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import re
import os
import skimage.io as io


def _build_dictionary(vqa, min_thre=6):
    """
    :param vqa: VQA instance
    :param min_thre: minimal times for a word ta appearing in the dataset
    :return: word dictionary list
    """
    counter = {}
    for q in vqa.qqa.items()[:10]:
        print(q)
        print(q[1]['question'])
        words = re.findall(r"[\w']+", q[1]['question'])
        for word in words:
            if counter[word]:
                counter[word] += 1
            else:
                counter[word] = 1
    d = {}
    indx = 0
    for word, num in counter.items():
        if num > min_thre:
            d[word] = indx
            indx += 1
    return d



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
        if True:
            self.dictionary = {}  # TODO: load from disk
        else:
            self.dictionary = _build_dictionary(self.vqa)

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, idx):
        ann = self.vqa.loadQA(idx)[0]
        image_id = ann['image_id']
        image_path = os.path.join(self.image_dir, self.image_filename_pattern.format(image_id))
        image = io.imread(image_path)


