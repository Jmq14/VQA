import torch.nn as nn


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image, question_encoding):
        # TODO
        raise NotImplementedError()
