import torch.nn as nn
import torch.nn.functional as F
from external.googlenet.googlenet import googlenet
import torch


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, n_ques, n_ans):
        super().__init__()
        self.img_feat = googlenet(pretrained=True)
        self.ques_feat = nn.Linear(n_ques, 1024)
        self.fc = nn.Linear(2048, n_ans+1)

    def forward(self, image, question_encoding):
        question_encoding = torch.max(question_encoding, 1)[0]
        img_feat = self.img_feat(image)
        ques_feat = self.ques_feat(question_encoding)
        feat = torch.cat((img_feat, ques_feat), 1)
        # print(img_feat.shape, ques_feat.shape, feat.shape)
        return self.fc(feat)

