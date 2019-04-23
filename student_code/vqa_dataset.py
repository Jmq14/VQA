from torch.utils.data import Dataset
from external.vqa.vqa import VQA
import re
import os
# import skimage.io as io
from PIL import Image
import numpy as np
import collections
import pickle
import torch


def _get_majority_ans(answers):
    answers = list(map(lambda x: x['answer'], answers))
    counter = collections.Counter(answers)
    majority = counter.most_common()[0][0]
    return majority


def _build_question_dictionary(vqa, min_thre=0):
    """
    :param vqa: VQA instance
    :param min_thre: only words that occur more than this number of times will be put in vocab
    :return: word-index dictionary
    """
    counter = collections.defaultdict(int)
    for i, q in vqa.qqa.items():
        words = re.findall(r"[\w']+", q['question'])
        for word in words:
            counter[word] += 1
    question_dict = {}
    indx = 0
    for word, num in counter.items():
        if num > min_thre:
            question_dict[word] = indx
            indx += 1
    return question_dict


def _build_answer_dictionary(vqa, min_thre=0):
    """
    :param vqa: VQA instance
    :param min_thre: minimal times for an answer appearing in the dataset
    :return: answer sequence - index dictionary
    """
    counter = collections.defaultdict(int)
    for ques_idx in vqa.getQuesIds():
        answers = vqa.qa[ques_idx]['answers']
        answer = _get_majority_ans(answers)
        counter[answer] += 1
    ans_dict = {}
    indx = 0
    for ans, num in counter.items():
        if num > min_thre:
            ans_dict[ans] = indx
            indx += 1
    return ans_dict


def _encode_question(sentence, dictionary, max_question_length=26):
    """
    :param sentence: question sentence
    :param dictionary: word - index dictionary
    :param max_question_length: max length of a caption, in number of words. captions longer than this get clipped
    :return: M x N one-hot torch tensor (M is the max number of words; N is the number of vocabularies)
    """
    words = re.findall(r"[\w']+", sentence)
    encode = torch.zeros((max_question_length, len(dictionary))).type(torch.FloatTensor)
    for i, word in enumerate(words):
        if i >= max_question_length:
            break
        if word in dictionary.keys():
            encode[i, dictionary[word]] = 1
    return encode


def _encode_answer(sentence, dictionary):
    """
    :param sentence: answer sentence
    :param: answer - index dictionary
    :return: indices
    """
    # encode = torch.zeros((len(dictionary) + 1)).type(torch.LongTensor)
    idx = len(dictionary) 
    if sentence in dictionary.keys():
        idx = dictionary[sentence]
    return idx


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 is_training=True, transform=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self.vqa = VQA(annotation_json_file_path, question_json_file_path)
        self.ques_idx_list = self.vqa.getQuesIds()
        self.image_dir = image_dir
        self.image_filename_pattern = image_filename_pattern

        if os.path.exists('ques_dictionary.pkl'):
            with open('ques_dictionary.pkl', 'rb') as f:
                self.dictionary = pickle.load(f) 
        else:
            if is_training:
                self.dictionary = _build_question_dictionary(self.vqa)
                with open('ques_dictionary.pkl', 'wb') as f:
                    pickle.dump(self.dictionary, f)
            else:
                raise "No dictionary built from training dataset!"

        if os.path.exists('ans_dictionary.pkl'): 
            with open('ans_dictionary.pkl', 'rb') as f:
                self.answers = pickle.load(f) 
        else:
            if is_training:
                self.answers = _build_answer_dictionary(self.vqa)
                with open('ans_dictionary.pkl', 'wb') as f:
                    pickle.dump(self.answers, f)
            else:
                raise "No answer list built from training dataset!"

        # print(self.dictionary)
        # print(self.answers)
        self.image_transform = transform

    def __len__(self):
        return len(self.ques_idx_list)

    def __getitem__(self, idx):
        ques_idx = self.ques_idx_list[idx]
        ann = self.vqa.loadQA(ques_idx)[0]

        image_id = ann['image_id']
        image_name = self.image_filename_pattern.format(str(image_id).zfill(12))
        image_path = os.path.join(self.image_dir, image_name)
        if os.path.splitext(image_path)[1] == '.npy':
            image = np.load(image_path).T
        else:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)

        question = self.vqa.qqa[ques_idx]['question']
        answers = ann['answers']
        best_answer = _get_majority_ans(answers)
        return {
                'image': image,
                'image_path': image_name,
                'question': question,
                'answer': best_answer,
                'question_encoding': _encode_question(question, self.dictionary),
                'answer_encoding': _encode_answer(best_answer, self.answers),
                }


