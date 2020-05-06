import config
import models
import os
import argparse
import torch

torch.set_printoptions(profile="full")      # print full debug
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # assign available GPUs
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='shtcnn', help='name of the model')
args = parser.parse_args()
model = {
    'pcnn_att': models.PCNN_ATT,
    'pcnn_one': models.PCNN_ONE,
    'pcnn_ave': models.PCNN_AVE,
    'cnn_att': models.CNN_ATT,
    'cnn_one': models.CNN_ONE,
    'cnn_ave': models.CNN_AVE,
    'rnn_att': models.RNN_ATT,
    'fads': models.FADS,
    'shtcnn': models.SHTCNN,
    'head_tail': models.HeadTailCNN_ATT
}
con = config.Config()
con.set_max_epoch(15)
print(con.max_epoch)
con.load_train_data()
con.load_test_data()
con.set_train_model(model[args.model_name])
# con.bi_train()    # train binary classifier
con.train()     # joint train binary classifier and multiclass classifier
