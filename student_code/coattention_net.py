import torch
import torch.nn as nn
import torch.nn.functional as F

"""
N: batch size
S: length of sequence (26)
"""


class AttentionNet(nn.Module):
    def __init__(self, n_feat, n_hidden):
        super().__init__()

        self.feature_head = nn.Linear(n_feat, n_hidden)
        self.attention_head = nn.Linear(n_feat, n_hidden)
        self.activate = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.predict = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, X, g=None):
        feat = self.feature_head(X)
        if g is not None:
            feat += self.attention_head(g)

        feat = self.activate(feat)
        w = self.predict(feat)
        att_x = torch.sum(w * X, dim=1, keepdim=True)

        return att_x  # N x 1 x n_feat


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, n_img, n_ques, n_ans, n_emb):
        super().__init__()
        self.word_level = nn.Sequential(
            nn.Linear(n_ques, n_emb),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.phase_level_1 = nn.Conv1d(n_emb, n_emb, kernel_size=1, padding=0)
        self.phase_level_2 = nn.Conv1d(n_emb, n_emb, kernel_size=2, padding=0)
        self.phase_level_3 = nn.Conv1d(n_emb, n_emb, kernel_size=3, padding=1)
        self.phase_level_activate = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.ques_level = nn.LSTM(input_size=n_emb, hidden_size=n_emb, batch_first=True)

        self.image_encoder = nn.Sequential(
            nn.Linear(n_img, n_emb),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.question_attention = AttentionNet(n_emb, n_hidden=512)
        self.image_attention = AttentionNet(n_emb, n_hidden=512)

        self.word_level_fuse = nn.Sequential(
            nn.Linear(n_emb, n_emb),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.phrase_level_fuse = nn.Sequential(
            nn.Linear(n_emb * 2, n_emb),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )
        self.ques_level_fuse = nn.Sequential(
            nn.Linear(n_emb * 2, n_emb * 2),
            nn.Tanh(),
            nn.Dropout(p=0.5)
        )

        self.predict = nn.Linear(n_emb * 2, n_ans)
        # softmax is implemented in loss function

    def _alternating_co_attention(self, Q, V):
        s = self.question_attention(Q, None)
        v = self.image_attention(V, s)
        q = self.question_attention(Q, v)
        return v.squeeze(), q.squeeze()  # N x n_feat

    def forward(self, image, question_encoding):

        # ============= Question Hierarchy  =============
        # word level
        word_feat = self.word_level(question_encoding)  # N x S x n_emb

        # phase level
        word_feat_T = word_feat.permute(0, 2, 1)  # N x n_emb x S
        phase_feat_1 = self.phase_level_1(word_feat_T)
        phase_feat_2 = self.phase_level_2(F.pad(word_feat_T, (0, 1)))
        phase_feat_3 = self.phase_level_3(word_feat_T)
        phase_feat_T = torch.max(torch.cat([
            phase_feat_1[:, :, :, None],
            phase_feat_2[:, :, :, None],
            phase_feat_3[:, :, :, None]], 3), 3, keepdim=False)[0]
        phase_feat = phase_feat_T.permute(0, 2, 1)  # N x S x n_emb
        # print(phase_feat_1.shape, phase_feat_2.shape, phase_feat_3.shape, phase_feat.shape)
        phase_feat = self.phase_level_activate(phase_feat)

        # question level
        # print(phase_feat.shape)
        ques_feat, _ = self.ques_level(phase_feat)  # N x S x n_emb

        # ============= Alter Attention  =============
        img_embed = self.image_encoder(image)

        v_w, q_w = self._alternating_co_attention(word_feat, img_embed)
        v_p, q_p = self._alternating_co_attention(phase_feat, img_embed)
        v_s, q_s = self._alternating_co_attention(ques_feat, img_embed)

        # -- fuse
        word_att_feat = self.word_level_fuse(q_w + v_w)

        phase_att_feat = torch.cat([q_p + v_p, word_att_feat], dim=1)
        phrase_att_feat = self.phrase_level_fuse(phase_att_feat)

        ques_att_feat = torch.cat([q_s + v_s, phrase_att_feat], dim=1)
        ques_att_feat = self.ques_level_fuse(ques_att_feat)

        output = self.predict(ques_att_feat)

        return output

