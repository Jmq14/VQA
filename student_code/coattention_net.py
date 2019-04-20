import torch.nn as nn


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()

    def forward(self, image, question_encoding):
        # TODO
        raise NotImplementedError()
