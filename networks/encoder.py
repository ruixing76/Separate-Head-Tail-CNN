import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _CNN(nn.Module):


    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_length 
        self.in_width = self.config.word_size + 2 * self.config.pos_size  
        self.kernel_size = (self.config.window_size, self.in_width) 
        self.out_channels = self.config.hidden_size
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, embedding):
        return self.cnn(embedding)


class _PiecewisePooling(nn.Module):
    def __init(self):
        super(_PiecewisePooling, self).__init__()

    def forward(self, x, mask, hidden_size):
        mask = torch.unsqueeze(mask, 1)
        x, _ = torch.max(mask + x, dim=2)
        x = x - 100
        return x.view(-1, hidden_size * 3)


class _MaxPooling(nn.Module):


    def __init__(self):
        super(_MaxPooling, self).__init__()

    def forward(self, x, hidden_size):
        # print(x.shape)  # torch.Size([244, 230, 120, 1])
        # print('x mean:', torch.mean(x))
        x, _ = torch.max(x, dim=2)
        # print(x.shape)  # torch.Size([244, 230, 1])
        return x.view(-1, hidden_size)


class PCNN(nn.Module):


    def __init__(self, config):
        super(PCNN, self).__init__()
        self.config = config
        self.mask = None
        self.cnn = _CNN(config)
        self.pooling = _PiecewisePooling()
        self.activation = nn.ReLU()
        self.batch_sen_len = None

    def forward(self, embedding):
        self.mask = self.config.mask
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.mask, self.config.hidden_size)
        return self.activation(x)


class CNN(nn.Module):


    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn = _CNN(config)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()
        self.batch_sen_len = None

    def forward(self, embedding):
        # batch*channel*height*width
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.config.hidden_size)
        return self.activation(x)


"""
Subject Object Embedding
"""


class _SOCNN(nn.Module):

    def __init__(self, config, is_head=True):
        super(_SOCNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.kernel_size = (3, self.config.model_dim)  
        self.out_channels = self.config.hidden_size
        self.padding = (1, 0)
        self.stride = 1
        self.batch_sen_len = None

        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, padding=self.padding)

        self.pooling = _MPiecewisePooling(is_head)
        self.activation = nn.ReLU()

    def forward(self, embedding):
        self.mask = self.config.mask
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.mask, self.config.hidden_size)
        return self.activation(x)


class _MPiecewisePooling(nn.Module):
    def __init__(self, is_head):
        super(_MPiecewisePooling, self).__init__()
        self.is_head = is_head

    def forward(self, x, mask, hidden_size):
        if self.is_head is True:
            part_mask = torch.stack((mask[:, :, 0], mask[:, :, 1] + mask[:, :, 2]), dim=0)
        else:
            part_mask = torch.stack((mask[:, :, 0] + mask[:, :, 1], mask[:, :, 2]), dim=0)
        part_mask = part_mask.permute(1, 2, 0)

        part_mask = torch.unsqueeze(part_mask, 1)
        x, _ = torch.max(part_mask + x, dim=2)
        x = x - 100
        return x.view(-1, hidden_size * 2)


class HeadTailCNN(nn.Module):
    def __init__(self, config):
        super(HeadTailCNN, self).__init__()
        self.config = config
        self.mask = None

        self.subject_cnn = _SOCNN(config, is_head=True)  # subject cnn
        self.object_cnn = _SOCNN(config, is_head=False)  # object cnn

        self.soc_ffn1 = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size * 3)
        self.soc_ffn2 = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size * 2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(self.config.drop_prob)

    def forward(self, embedding):
        sbj_out = self.subject_cnn(embedding)
        obj_out = self.object_cnn(embedding)

        soc = torch.cat((sbj_out, obj_out), dim=1)
        soc = self.relu(self.soc_ffn1(soc))
        soc_embedding = self.relu(self.soc_ffn2(soc))
        # soc_embedding=soc

        return soc_embedding


class ContextRNN(nn.Module):
    def __init__(self, config):
        super(ContextRNN, self).__init__()
        self.config = config
        self.batch_sen_len = None
        self.gru = nn.GRU(input_size=self.config.word_size + 2 * self.config.pos_size,
                          hidden_size=self.config.hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.ffn = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(self, embedding, batch_len=None):
        lengths_sorted, sorted_idx = torch.sort(batch_len, descending=True)
        embedding = embedding[sorted_idx]
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, lengths_sorted, batch_first=True)
        _, hidden = self.gru(batch_packed, None)
        _, desorted_indices = torch.sort(sorted_idx, descending=False)
        # hidden = hidden.squeeze(0)[desorted_indices]
        # output = torch.cat((hidden[0], hidden[1]), dim=1)
        output = hidden[0] + hidden[1]
        output = output[desorted_indices]
        # output=F.relu(self.ffn(output))
        return output


class SemanticEmbedding(nn.Module):
    def __init__(self, config):
        super(SemanticEmbedding, self).__init__()
        self.config = config
        self.mask = None
        self.batch_sen_len = None
        self.subject_cnn = _SOCNN(config, is_head=True)  # subject cnn
        self.object_cnn = _SOCNN(config, is_head=False)  # object cnn

        self.context_cnn = PCNN(config)  # context cnn
        self.context_gru = ContextRNN(config)

        self.soc_ffn1 = nn.Linear(self.config.hidden_size * 4, self.config.hidden_size * 3)
        self.soc_ffn2 = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)

        self.context_ffn = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(self.config.drop_prob)

    def forward(self, embedding):
        sbj_out = self.subject_cnn(embedding)
        obj_out = self.object_cnn(embedding)

        context_out = self.context_cnn(embedding)                           # Context CNN
        # context_out = self.context_gru(embedding,self.batch_sen_len)      # Context RNN

        soc = torch.cat((sbj_out, obj_out), dim=1)
        soc = self.relu(self.soc_ffn1(soc))
        soc_embedding = self.relu(self.soc_ffn2(soc))

        context_embedding = self.relu(self.context_ffn(context_out))

        # soc_embedding = self.dropout(output)
        # context=self.dropout(context)
        # return soc_embedding, context_embedding
        return soc_embedding, context_embedding