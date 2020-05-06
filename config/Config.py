# coding:utf-8
# author: Rui Xing
import torch
import torch.optim as optim
import numpy as np
import os
import datetime
import sys
import sklearn.metrics
from tqdm import tqdm


def to_var(x, is_cuda=True):
    """
    Change to Tensor
    """
    if is_cuda is True:
        return torch.from_numpy(x).cuda()
    else:
        return torch.from_numpy(x)


class Accuracy(object):
    """
    Use a single class for Accuracy
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Config(object):
    def __init__(self):
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.binary_not_NA_acc = Accuracy()
        self.binary_NA_acc = Accuracy()

        self.data_path = './data'
        self.use_bag = True
        self.use_gpu = True
        self.is_training = True
        self.max_length = 120
        self.pos_num = 2 * self.max_length  # two positions
        self.num_classes = 53
        self.hidden_size = 230
        self.pos_size = 5  
        self.max_epoch = 15
        self.opt_method = 'sgd'
        self.optimizer = None

        self.learning_rate = 0.5
        self.weight_decay = 1e-5

        self.drop_prob = 0.5
        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_res'
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.batch_size = 160
        self.word_size = 50
        self.window_size = 3
        self.epoch_range = None

        self.model_dim = 60
        self.num_heads = 5

        self.mask = None
        self.flag = 'multi'  # flag of network for binary/multi classification


    def load_train_data(self):
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_train_word = np.load(os.path.join(self.data_path, 'train_word.npy'))  
        self.data_train_pos1 = np.load(os.path.join(self.data_path, 'train_pos1.npy'))  
        self.data_train_pos2 = np.load(os.path.join(self.data_path, 'train_pos2.npy'))  
        self.data_train_entity_pos = np.load(os.path.join(self.data_path, 'train_entity_pos.npy'))  

        self.data_train_mask = np.load(os.path.join(self.data_path, 'train_mask.npy'))  
        self.data_train_sen_len = np.load(os.path.join(self.data_path, 'train_sen_len.npy'))  


        if self.use_bag:
            self.data_query_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))  
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_bag_label.npy'))  
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_bag_scope.npy'))  # instance scope 
        else:
            self.data_train_label = np.load(os.path.join(self.data_path, 'train_ins_label.npy'))
            self.data_train_scope = np.load(os.path.join(self.data_path, 'train_ins_scope.npy'))
        print("Finish reading")
        self.train_order = list(range(len(self.data_train_label)))  # training order
        self.train_batches = int(len(self.data_train_label) / self.batch_size)  # batch number
        if len(self.data_train_label) % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_test_word = np.load(os.path.join(self.data_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(self.data_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(self.data_path, 'test_pos2.npy'))
        self.data_test_entity_pos = np.load(os.path.join(self.data_path, 'test_entity_pos.npy'))  # entity position
        self.data_test_mask = np.load(os.path.join(self.data_path, 'test_mask.npy'))
        self.data_test_sen_len = np.load(os.path.join(self.data_path, 'test_sen_len.npy'))  # sentence length

        if self.use_bag:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_bag_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_bag_scope.npy'))
        else:
            self.data_test_label = np.load(os.path.join(self.data_path, 'test_ins_label.npy'))
            self.data_test_scope = np.load(os.path.join(self.data_path, 'test_ins_scope.npy'))
        print("Finish reading")
        self.test_batches = int(len(self.data_test_label) / self.batch_size)
        if len(self.data_test_label) % self.batch_size != 0:
            self.test_batches += 1

        self.total_recall = self.data_test_label[:, 1:].sum()

    def set_train_model(self, model):
        """
        set training models
        :param model:
        :return:
        """
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)  
        if self.pretrain_model != None:
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        self.trainModel.cuda()
        # Set optimizer
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.learning_rate, lr_decay=self.lr_decay,
                                           weight_decay=self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.learning_rate,
                                            weight_decay=self.weight_decay)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        print("Finish initializing")

    def set_test_model(self, model):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def get_train_batch(self, batch):
        """
        get training batch information,save to global for training
        :param batch: batch id
        :return:
        """
        
        input_scope = np.take(self.data_train_scope,
                              self.train_order[batch * self.batch_size: (batch + 1) * self.batch_size], axis=0)
        index = []  
        scope = [0]
        
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))  
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        
        self.batch_word = self.data_train_word[index, :]  
        self.batch_pos1 = self.data_train_pos1[index, :]  
        self.batch_pos2 = self.data_train_pos2[index, :]  
        self.batch_entity_pos=self.data_train_entity_pos[index,:]
        self.batch_mask = self.data_train_mask[index, :]  
        self.batch_sen_len = self.data_train_sen_len[index]  

        
        self.batch_label = np.take(self.data_train_label,
                                   self.train_order[batch * self.batch_size: (batch + 1) * self.batch_size], axis=0)
        self.batch_attention_query = self.data_query_label[index]  
        self.batch_scope = scope  

    def get_test_batch(self, batch):
        """
        get test batch information,save to global for testing
        :param batch:
        :return:
        """
        input_scope = self.data_test_scope[batch * self.batch_size: (batch + 1) * self.batch_size]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_test_word[index, :]
        self.batch_pos1 = self.data_test_pos1[index, :]
        self.batch_pos2 = self.data_test_pos2[index, :]
        self.batch_entity_pos = self.data_test_entity_pos[index, :]
        self.batch_mask = self.data_test_mask[index, :]
        self.batch_sen_len = self.data_test_sen_len[index]  
        self.batch_scope = scope

        self.batch_label = self.data_test_label[batch * self.batch_size: (batch + 1) * self.batch_size, :]

    def train_one_step(self):
        """
        train one step
        :return:
        """
        self.trainModel.embedding.word = to_var(self.batch_word)  
        self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
        self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
        self.trainModel.embedding.entity_pos=to_var(self.batch_entity_pos)
        self.trainModel.embedding.batch_sen_len = to_var(self.batch_sen_len)

        self.trainModel.encoder.mask = to_var(self.batch_mask)
        self.trainModel.encoder.batch_sen_len = to_var(self.batch_sen_len)
        self.mask = to_var(self.batch_mask)

        self.trainModel.selector.scope = self.batch_scope
        self.trainModel.selector.attention_query = to_var(self.batch_attention_query)  
        self.trainModel.selector.binary_attention_query = self.trainModel.selector.attention_query.clone()
        self.trainModel.selector.binary_attention_query[self.trainModel.selector.binary_attention_query != 0] = 1   

        self.trainModel.selector.label = to_var(self.batch_label)
        self.trainModel.selector.binary_label = self.trainModel.selector.label.clone()
        self.trainModel.selector.binary_label[self.trainModel.selector.binary_label!=0]=1
        self.trainModel.classifier.label = to_var(self.batch_label)

        self.optimizer.zero_grad()  
        loss, _output = self.trainModel()  
        if loss != 0:
            loss.backward()  
            self.optimizer.step()
        # calculate loss for binary classification
        if self.flag == 'binary':
            for i, prediction in enumerate(_output):
                # ground_truth = 0 if self.batch_label[i] == 0 else 1
                ground_truth = self.batch_label[i]
                if ground_truth == 0:
                    self.binary_NA_acc.add(prediction == ground_truth)
                if ground_truth == 1:
                    self.binary_not_NA_acc.add(prediction == ground_truth)
                # print('prediction: %d,ground_truth %d\n'% (prediction.data,ground_truth))
            if loss != 0:
                return loss.item()
            else:
                return loss
        # calculate loss for multi classification
        elif self.flag == 'multi':
            for i, prediction in enumerate(_output):
                if self.batch_label[i] == 0:
                    self.acc_NA.add(prediction == self.batch_label[i])
                else:
                    self.acc_not_NA.add(prediction == self.batch_label[i])
                self.acc_total.add(prediction == self.batch_label[i])
            if loss != 0:
                return loss.item()
            else:
                return loss

    def test_one_step(self):
        self.testModel.embedding.word = to_var(self.batch_word)
        self.testModel.embedding.pos1 = to_var(self.batch_pos1)
        self.testModel.embedding.pos2 = to_var(self.batch_pos2)
        self.testModel.embedding.entity_pos = to_var(self.batch_entity_pos)
        self.testModel.embedding.batch_sen_len = to_var(self.batch_sen_len)
        self.testModel.encoder.batch_sen_len = to_var(self.batch_sen_len)

        self.testModel.encoder.mask = to_var(self.batch_mask)
        self.mask = to_var(self.batch_mask)
        self.testModel.selector.scope = self.batch_scope
        self.testModel.selector.label = to_var(self.batch_label)
        self.testModel.classifier.label = to_var(self.batch_label)
        return self.testModel.test()

    def bi_train(self):
        """
        binary classification
        :return:
        """
        self.flag = 'binary'
        best_auc = 0
        bi_path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-binary')
        if os.path.exists(bi_path):
            print('loading binary classifier parameters')
            self.trainModel.load_state_dict(torch.load(bi_path))
            self.testModel = self.trainModel
            f1_score, auc = self.bi_test_one_epoch()
            best_auc = auc

        print('Start training binary classifier...')
        for epoch in range(self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            self.binary_not_NA_acc.clear()  
            self.binary_NA_acc.clear()
            np.random.shuffle(self.train_order)

            for batch in range(self.train_batches):

                self.get_train_batch(batch)
                loss = self.train_one_step()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write(
                    "epoch %d step %d time %s | loss: %f, not NA accuracy: %f | NA accuracy %f\r" % (
                        epoch, batch, time_str, loss, self.binary_not_NA_acc.get(), self.binary_NA_acc.get()))
                sys.stdout.flush()

            if (epoch + 1) % self.test_epoch == 0:
                self.testModel = self.trainModel
                f1_score, auc = self.bi_test_one_epoch()
                if auc > best_auc:
                    best_auc = auc
                    print('Saving binary model...')
                    path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-binary')
                    torch.save(self.trainModel.state_dict(), path)
                    print('Have saved model to ' + path)
        print("Finish training binary classifier.")

    def train(self):
        """
        multi classification
        :return:
        """
        self.flag = 'multi'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_auc = 0.0
        best_p = None
        best_r = None
        best_epoch = 0
        bi_path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-binary')
        if os.path.exists(bi_path):
            print('loading binary classifier parameters')
            self.trainModel.load_state_dict(torch.load(bi_path))


        for epoch in range(self.max_epoch):
            print('Epoch ' + str(epoch) + ' starts...')
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()
            # first shuffle
            np.random.shuffle(self.train_order)
            for batch in range(self.train_batches):
                # train
                self.get_train_batch(batch)
                loss = self.train_one_step()
                # print, use datetime to check
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write(
                    "epoch %d step %d time %s | loss: %f, NA accuracy: %f, not NA accuracy: %f, total accuracy: %f\r" % (
                        epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
                sys.stdout.flush()
            # save
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print('Have saved model to ' + path)
            if (epoch + 1) % self.test_epoch == 0:
                self.testModel = self.trainModel
                auc, pr_x, pr_y = self.test_one_epoch()
                if auc > best_auc:
                    best_auc = auc
                    best_p = pr_x
                    best_r = pr_y
                    best_epoch = epoch
        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def bi_test_one_epoch(self):
        label = list()
        for each in self.data_test_label:
            if each[0] == 1:
                label.append(0)
            else:
                label.append(1)
        test_predict = []
        test_score = []
        for batch in tqdm(range(self.test_batches)):
            self.get_test_batch(batch)
            batch_score, binary_softmax = self.test_one_step()
            test_predict = test_predict + batch_score
            test_score = test_score + binary_softmax
        
        tp = fp = tn = fn = 0
        for i in range(len(test_predict)):
            if test_predict[i] == 1 and label[i] == 1:
                tp += 1
            elif test_predict[i] == 1 and label[i] == 0:
                fp += 1
            elif test_predict[i] == 0 and label[i] == 0:
                tn += 1
            elif test_predict[i] == 0 and label[i] == 1:
                fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (2 * precision * recall) / (precision + recall)
        print('predict positive num:%f\n' % (tp + fp))
        print('truly positive:%f\n' % (tp + fn))
        print("positive precision:%f \n" % precision)
        print("positive recall:%f \n" % recall)
        print("F1 Score:%f\n" % f1_score)

        # AUC calculation
        binary_label = list()
        for each in self.data_test_label:
            if each[0] == 1:
                binary_label.append([1, 0])
            else:
                binary_label.append([0, 1])
        test_result = []
        for i in range(len(test_score)):
            test_result.append([binary_label[i][1], test_score[i][1]])  
        
        test_result = sorted(test_result, key=lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  # precision
            pr_x.append(float(correct) / self.total_recall)  # recall
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print("auc: ", auc)
        return f1_score, auc

    def test_one_epoch(self):
        test_score = []
        for batch in tqdm(range(self.test_batches)):
            self.get_test_batch(batch)
            batch_score = self.test_one_step()  
            test_score = test_score + batch_score
        test_result = []
        for i in range(len(test_score)):
            for j in range(1, len(test_score[i])):  
                test_result.append([self.data_test_label[i][j], test_score[i][j]])  
        
        test_result = sorted(test_result, key=lambda x: x[1])
        test_result = test_result[::-1]
        pr_x = []
        pr_y = []
        correct = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  # precision
            pr_x.append(float(correct) / self.total_recall)  # recall
        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        print("auc: ", auc)
        print("precision P@100:{} |P@300:{} |P@500:{}|P@700:{}|P@1000:{}|P@1761:{} ".format(pr_y[100], pr_y[300],
                                                                                            pr_y[500], pr_y[700],
                                                                                            pr_y[1000], pr_y[1761]))
        print(
            "recall R@100:{} |R@300:{} |R@500:{}|R@700:{}|R@1000:{}|R@1761:{} ".format(pr_x[100], pr_x[300], pr_x[500],
                                                                                       pr_x[700], pr_x[1000],
                                                                                       pr_x[1761]))
        return auc, pr_x, pr_y

    def test(self):
        best_epoch = None
        best_auc = 0.0
        best_p = None
        best_r = None
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            print(path)
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d" % (epoch))
            # test binary network
            self.testModel.load_state_dict(torch.load(path))
            self.flag = 'binary'
            self.bi_test_one_epoch()
            self.flag = 'multi'
            auc, p, r = self.test_one_epoch()
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                best_p = p
                best_r = r
            print("Finish testing epoch %d" % (epoch))
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), best_p)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), best_r)
        print("Finish storing")

    def bi_test(self):
        bi_path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-binary')
        if os.path.exists(bi_path):
            print('loading binary classifier parameters')
            self.testModel.load_state_dict(torch.load(bi_path))
        self.flag = 'binary'
        self.bi_test_one_epoch()
