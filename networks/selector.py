import torch
import torch.nn as nn
import torch.nn.functional as F


class Selector(nn.Module):
    """
    merge sentence to bag representations
    """

    def __init__(self, config, relation_dim):
        super(Selector, self).__init__()
        self.config = config
        self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)  
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))  
        self.attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)

        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.label = None
        self.dropout = nn.Dropout(self.config.drop_prob)

    def init_weights(self):
        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)
        nn.init.xavier_uniform_(self.attention_matrix.weight.data)

    def get_logits(self, x):
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1), ) + self.bias
        return logits

    def forward(self, x):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError


class Attention(Selector):
    def _attention_train_logit(self, x):
        relation_query = self.relation_matrix(self.attention_query)
        attention = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * attention * relation_query, 1, True)
        return attention_logit

    def _attention_test_logit(self, x):
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0,
                                                          1))
        return attention_logit

    def forward(self, x):
        attention_logit = self._attention_train_logit(x)
        tower_repre = []

        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        attention_logit = self._attention_test_logit(x)
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)

        # _, test_label = torch.max(self.label, dim=1)
        # test_loss = self.test_loss(stack_output, test_label)
        # print('test_loss:%f' % (test_loss))
        return list(stack_output.data.cpu().numpy())


class One(Selector):
    def forward(self, x):
        tower_logits = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]] 
            sen_matrix = self.dropout(sen_matrix)
            logits = self.get_logits(sen_matrix)  
            score = F.softmax(logits, dim=1)  
            _, k = torch.max(score, dim=0) 
            # print(k)
            k = k[self.label[i]]  
            tower_logits.append(logits[k])
        return torch.stack(tower_logits, 0)

    def test(self, x):
        tower_score = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            logits = self.get_logits(sen_matrix)
            score = F.softmax(logits, 1)
            score, _ = torch.max(score, 0)
            tower_score.append(score)
        tower_score = torch.stack(tower_score)
        return list(tower_score.data.cpu().numpy())


class Average(Selector):
    def forward(self, x):
        # print('selector:',x.size())
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        # print('stack_repre size',stack_repre.size())
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        # print('logits size', stack_repre.size())
        return logits

    def test(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        logits = self.get_logits(stack_repre)
        score = F.softmax(logits, 1)

        test_loss_criterion = nn.CrossEntropyLoss()
        _, test_label = torch.max(self.label, dim=1)
        _, predict = torch.max(logits, dim=1)
        # print('test label\n', test_label)
        # print('predict\n', predict)
        test_loss = test_loss_criterion(logits, test_label)
        # print('test_loss:%f' % (test_loss))
        true = test_label != predict
        acc = torch.sum(test_label != predict)
        # print('true:\n', true)
        # print('acc:\n', acc)

        return list(score.data.cpu().numpy())


class TwoStageSelector(nn.Module):
    """
    Attention Layer for Selecting Bag Representation
    """

    def __init__(self, config, relation_dim):
        super(TwoStageSelector, self).__init__()
        self.config = config
        # Binary Classification Attention
        self.binary_relation_matrix = nn.Embedding(2, relation_dim)  
        self.binary_bias = nn.Parameter(torch.Tensor(2))  
        self.binary_attention_matrix = nn.Embedding(2, relation_dim)
        # Multi Classification Attention
        self.multi_relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)  
        self.multi_bias = nn.Parameter(torch.Tensor(self.config.num_classes))  
        self.multi_attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)

        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.binary_attention_query = None
        self.label = None
        self.binary_label = None
        self.dropout = nn.Dropout(self.config.drop_prob)

    def init_weights(self):
        nn.init.xavier_uniform_(self.multi_relation_matrix.weight.data)
        nn.init.normal_(self.multi_bias)
        nn.init.xavier_uniform_(self.multi_attention_matrix.weight.data)

        nn.init.xavier_uniform_(self.binary_relation_matrix.weight.data)
        nn.init.normal_(self.binary_bias)
        nn.init.xavier_uniform_(self.binary_attention_matrix.weight.data)

    def get_logits(self, x, is_multi=True):
        """
        :param is_multi:
        :param x:
        :return:
        """
        if is_multi is True:
            relation_matrix = self.multi_relation_matrix
            bias = self.multi_bias
        else:
            relation_matrix = self.binary_relation_matrix
            bias = self.binary_bias

        logits = torch.matmul(x, torch.transpose(relation_matrix.weight, 0, 1), ) + bias
        return logits

    def _attention_train_logit(self, x, is_multi=True):
        if is_multi is True:
            attention_query = self.attention_query
            relation_matrix = self.multi_relation_matrix
            attention_matrix = self.multi_attention_matrix
        else:
            attention_query = self.binary_attention_query
            relation_matrix = self.binary_relation_matrix
            attention_matrix = self.binary_attention_matrix

        relation_query = relation_matrix(attention_query)
        attention = attention_matrix(attention_query)
        attention_logit = torch.sum(x * attention * relation_query, 1, True)
        return attention_logit

    def _attention_test_logit(self, x, is_multi=True):
        if is_multi is True:
            relation_matrix = self.multi_relation_matrix
            attention_matrix = self.multi_attention_matrix
        else:
            relation_matrix = self.binary_relation_matrix
            attention_matrix = self.binary_attention_matrix

        attention_logit = torch.matmul(x, torch.transpose(
            attention_matrix.weight * relation_matrix.weight, 0, 1))
        return attention_logit

    def attention_train(self, x, is_multi=True):
        attention_logit = self._attention_train_logit(x, is_multi)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), dim=1)
            final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre, is_multi)
        return logits

    def attention_test(self, x, is_multi=True):
        attention_logit = self._attention_test_logit(x, is_multi)
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]: self.scope[i + 1]], 0, 1), dim=1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre, is_multi)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)
        return stack_output


    def average_input(self, x, is_multi=True):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            soc_sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            soc_final_repre = torch.mean(soc_sen_matrix, 0)
            tower_repre.append(soc_final_repre)
        stack_repre = torch.stack(tower_repre)
        logits = self.get_logits(stack_repre,is_multi)
        # stack_repre = self.dropout(stack_repre)
        return logits

    def average_test(self, x, is_multi=True):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            soc_sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            soc_final_repre = torch.mean(soc_sen_matrix, 0)
            tower_repre.append(soc_final_repre)
        stack_repre = torch.stack(tower_repre)
        logits = self.get_logits(stack_repre,is_multi)
        score = F.softmax(logits, 1)
        return score

    def one_input(self, x, is_multi=True):
        """
        :param x: sen_embeddings
        :return:
        """
        tower_logits = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]  
            sen_matrix = self.dropout(sen_matrix)
            logits = self.get_logits(sen_matrix, is_multi)  
            score = F.softmax(logits, dim=1)  
            _, k = torch.max(score, dim=0)  
            k = k[self.binary_label[i]]  
            tower_logits.append(logits[k])
        return torch.stack(tower_logits, 0)

    def one_test(self, x, is_multi=True):
        tower_score = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]: self.scope[i + 1]]
            logits = self.get_logits(sen_matrix, is_multi)
            score = F.softmax(logits, 1)
            score, _ = torch.max(score, 0)
            tower_score.append(score)
        tower_score = torch.stack(tower_score)
        return tower_score

    def forward(self, soc_sen_embedding, context_sen_embedding):
        # Attention
        soc_bag_embedding = self.attention_train(soc_sen_embedding)
        context_bag_embedding = self.attention_train(context_sen_embedding, is_multi=False)

        # One
        # soc_bag_embedding = self.one_input(context_sen_embedding)
        # context_bag_embedding = self.one_input(context_sen_embedding)

        # Average
        # soc_bag_embedding = self.average_input(context_sen_embedding)
        # context_bag_embedding = self.average_input(context_sen_embedding,is_multi=False)

        return soc_bag_embedding, context_bag_embedding

    def test(self, soc_sen_embedding, context_sen_embedding):
        # Attention
        soc_bag_embedding = self.attention_test(soc_sen_embedding)
        context_bag_embedding = self.attention_test(context_sen_embedding,is_multi=False)

        # One
        # soc_bag_embedding = self.one_test(soc_sen_embedding)
        # context_bag_embedding = self.one_test(context_sen_embedding, is_multi=False)

        # Average
        soc_bag_embedding = self.average_test(soc_sen_embedding)
        context_bag_embedding = self.average_test(context_sen_embedding,is_multi=False)

        return soc_bag_embedding, context_bag_embedding
