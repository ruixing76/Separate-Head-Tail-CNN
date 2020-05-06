import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.label = None
        self.weight = torch.Tensor([0.1] + [1 for _ in range(52)])  # change weight
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits):
        loss = self.loss(logits, self.label)
        _, output = torch.max(logits, dim=1)
        return loss, output.data


class BinaryClassifier(nn.Module):
    """
    binary classifier
    """

    def __init__(self, config):
        super(BinaryClassifier, self).__init__()
        self.config = config
        self.ffn_binary = nn.Linear(self.config.hidden_size, 2, bias=True)

        self.weight = torch.Tensor([1, 1])  

        self.binary_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2)

        self.threshold = 0.1  

        self.alpha = torch.Tensor([0.25, 0.75]).view(2, 1)
        self.afocal_loss = AFocalLoss(class_num=2, alpha=self.alpha, gamma=2)

    def forward(self, bag_embedding, batch_label):
        # get binary label for auxiliary task
        label = batch_label
        binary_logits = bag_embedding

        binary_label = torch.where(label != 0, torch.full_like(label, 1), label)  # binary label
        binary_loss = self.binary_loss(binary_logits, binary_label)

        _, output = torch.max(binary_logits, dim=1)
        return binary_loss, output.data

    def inference(self, bag_embedding):
        """
        :param bag_embedding:
        :return:
        """
        # binary_logits = self.ffn_binary(bag_embedding)
        binary_logits = bag_embedding

        binary_softmax = F.softmax(binary_logits, dim=1)
        return binary_softmax

    def test(self, bag_embedding, label):
        binary_softmax = bag_embedding
        # binary_logits = self.ffn_binary(bag_embedding)
        # binary_softmax = F.softmax(binary_logits, dim=1)

        pos_score = binary_softmax[:, 1]
        pos_index = pos_score >= self.threshold
        _output = torch.full_like(pos_score, 0)
        _output[pos_index] = 1

        return _output, binary_softmax


class TwoStageClassifier(nn.Module):
    """
    two-stage classification
    joint training binary classifier and multi classifier
    """

    def __init__(self, config):
        super(TwoStageClassifier, self).__init__()
        self.config = config
        self.label = None

        self.ffn_multi = nn.Linear(self.config.hidden_size, 53)
        self.ffn_rel = nn.Linear(self.config.num_classes, 2)  
        # init.constant_(self.ffn_multi.bias.data[0], 4.6)
        # init.constant_(self.ffn_rel.bias.data[0], 4.6)
        self.binary_classifier = BinaryClassifier(config)
        self.threshold = 0.15  

        self.weight = torch.Tensor([1] + [1 for _ in range(52)])  
        self.multi_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()

    def forward(self, soc_bag_embedding, context_bag_embedding):
        label = self.label
        binary_loss = multi_loss = 0
        output = torch.Tensor(soc_bag_embedding.size(0))  
        if self.config.flag == 'binary':
            # train binary classifier
            binary_loss, output = self.binary_classifier(context_bag_embedding, label)
            return binary_loss, output.data
        elif self.config.flag == 'multi':
            binary_loss, _ = self.binary_classifier(context_bag_embedding, label)
            binary_softmax = self.binary_classifier.inference(context_bag_embedding)
            NA_index = [binary_softmax[:, 1] < self.threshold]
            not_NA_index = [binary_softmax[:, 1] >= self.threshold]
            not_NA_embedding = soc_bag_embedding[not_NA_index]
            not_NA_label = label[not_NA_index]
            output[NA_index] = 0

            if not_NA_embedding.size(0) != 0:
                # multi_logits = self.ffn_multi(not_NA_embedding)
                multi_logits = not_NA_embedding
                multi_loss = self.multi_loss(multi_logits, not_NA_label)
                _, max_index = torch.max(multi_logits, dim=1)  
                max_index = max_index.cpu().float()
                output.index_put_(not_NA_index, max_index)
            loss = multi_loss
            # print('binary loss %f, multi loss %f\n ' % (binary_loss, multi_loss))
            return loss, output.data

    def test(self, soc_bag_embedding, context_bag_embedding):
        _, test_label = torch.max(self.label, dim=1)
        if self.config.flag == 'binary':
            binary_predict, binary_softmax = self.binary_classifier.test(context_bag_embedding, test_label)
            return list(binary_predict.data.cpu().numpy()), list(binary_softmax.data.cpu().numpy())
        else:
            score = torch.cuda.FloatTensor(context_bag_embedding.size(0), self.config.num_classes)
            # get non NA
            # binary_softmax = self.binary_classifier.inference(context_bag_embedding)
            binary_softmax = context_bag_embedding
            # value, binary_label = torch.max(self.label, dim=1)
            # NA_index = [binary_label == 0]
            # not_NA_index = [binary_label != 0]
            NA_index = [binary_softmax[:, 1] < self.threshold]
            not_NA_index = [binary_softmax[:, 1] >= self.threshold]
            not_NA_embedding = soc_bag_embedding[not_NA_index]

            if not_NA_embedding.size(0) != 0:
                # multi_logits = self.ffn_multi(not_NA_embedding)
                multi_softmax = not_NA_embedding
                # max_value, index = torch.max(multi_softmax[:,1:], dim=1)
                # index=index+1
                # multi_softmax.scatter_(1, index.view(index.size(0), 1),
                #                        (relation_possibility[:, 1] * max_value).view(multi_softmax.size(0), 1))
                # print(multi_softmax)
                # multi_softmax = F.softmax(multi_logits, dim=1)
                score.index_put_(not_NA_index, multi_softmax)

            score.index_put_(NA_index,
                             torch.cuda.FloatTensor([0.99] + [1e-40 for _ in range(self.config.num_classes - 1)]))
            # softmax = F.softmax(score, dim=1)

            # debug
            # _, test_label = torch.max(self.label, dim=1)
            # _, predict = torch.max(softmax, dim=1)
            # test_loss = self.test_loss(score, test_label)
            # print('test label\n', test_label)
            # print('predict\n', predict)
            # print('test_loss:%f' % (test_loss))
            return list(score.data.cpu().numpy())

