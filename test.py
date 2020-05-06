import config
import models
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
con.load_test_data()
con.set_test_model(model[args.model_name])
con.set_epoch_range(list(range(4,8)))
# con.test()
con.bi_test()