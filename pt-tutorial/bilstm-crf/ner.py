# coding: utf-8
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
'''
http://59.80.44.45/www.cs.columbia.edu/~mcollins/crf.pdf
维特比算法
找出最有可能产生其观测序列的隐含序列
toy
'''
torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}


def argmax(vec):
    _, idx = torch.max(vec, 1)  # reduce vec, not matrix
    return idx.item()


def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    return torch.tensor(idx, dtype=torch.long)


def log_sum_exp(vec):
    '''vec - max, exp, 求和， log， 加最大值'''
    max_score = vec[0, argmax(vec)]  # 不是标量嘛
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # 为啥除以2.
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),  # (num_layers * num_directions, batch, hidden_size) batch 为 1
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()  # __init 函数里有了,每次个batch都把隐藏层参数随机初始化？
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)  # (seq_len, batch, input_size): 只有1个样本
        # lstm_out: (seq_len, batch, num_directions * hidden_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)  # batch去哪里了？
        lstm_feats = self.hidden2tag(lstm_out)  # 输出的维度是：len(sentence) * tagset_size
        return lstm_feats

    def _viterbi_decode(self, feats):
        # len(sentence) * tagset_size
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]  # self.transitions理解为到该状态的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)  # 标签id
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))  # 标签对应得分
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # cat 默认按行拼接
            # print("forward_var", forward_var)
            # print("torch.cat(viterbivars_t)", torch.cat(viterbivars_t))
            # print("viterbivars_t", viterbivars_t)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        print("backpointers", backpointers)
        print("feats: ", feats)
        print("transitions: ", self.transitions)
        print("terminal_var", terminal_var)
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)  # 理解为去max_score 加一个log
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):  # 为什么不正向计算呢, 转移概率正向反向不一样，b-i，跟i-b。此处理解的self.transitions i,j为从j到i
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)
]
# training_data = [
# (
#     "georgia tech is a university in georgia".split(),
#     "B I O O O O B".split()
# )
# ]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)  # 这个不错

# def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    # print(model(precheck_sent))


for epoch in range(1):
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags])
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))

'''ner model'''
