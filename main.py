import os, time, argparse, sys
from readline import add_history

import pandas as pd
import numpy as np
from torch.backends import cudnn
import torch

import warnings
warnings.filterwarnings('ignore')

from solver import Solver

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


def str2bool(v):
    return v.lower() in ('true')

DEBUG = True
# DEBUG = False

def main(config):
    cudnn.benchmark = True
    solver = Solver(config)

    if config.mode == 'train':
        # solver.begin() # for pu lp
        # solver.train() 
        # solver.test()
        solver.rank_test()
        # solver.pick_test()
    elif config.mode == 'test':
        solver.test()

    return solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--model_save_path', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # for Dilated CNN
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size in the dicnn')
    parser.add_argument('--output_size', default=64, type=int, help='output size in the dicnn')
    parser.add_argument('--kernel_size', default=2, type=int, help='kernel size of the conv layers')
    parser.add_argument('--n_layers', default=7, type=int, help='# of layers in the dicnn')
    parser.add_argument('--d_model', default=64, type=int, help='temporal encoding dimension')
    # parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

    # for WETAS Framework
    parser.add_argument('--pooling_type', default='avg', type=str, help='avg | max')
    parser.add_argument('--local_threshold', default=0.3, type=float, help='score threshold to identify anomalies')
    parser.add_argument('--granularity', default=4, type=int, help='granularity for sequential pseudo-labels')
    parser.add_argument('--beta', default=0.1, type=float, help='margin size for the alignment loss')
    parser.add_argument('--gamma', default=0.1, type=float, help='smoothing for differentiable DTW')
    
    # for Optimization
    # parser.add_argument('--batch_size', default=4353, type=int, help='batch size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--n_epochs', default=200, type=int, help="# of training epochs")
    parser.add_argument('--learning_rate', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--gpuidx', default=6, type=int, help='gpu index')
    parser.add_argument('--patience', default=3, type=int, help='# of patience for early stopping, 3 for gpt, 50 for others')
    parser.add_argument('--stopping', default='f1', type=str, help='f1 | loss')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=7, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')


    parser.add_argument('--dataset', default='EMG', type=str, help='EMG | MSL | SMD | SMAP')
    parser.add_argument('--win_size', default = 100, type=int, help='split size for preprocessing the data')
    parser.add_argument('--data_feature',default=8,type= int)
    parser.add_argument('--prior', default=0.25, type=float, help='class prior for the anomaly class')
    parser.add_argument('--anomaly_thre', default=0.0, type=float, help='threshold for classifying anomalies')
    parser.add_argument('--anomaly_ratio', default=0.65, type=float, help='point level ratio')
    parser.add_argument('--is_classifier', default='Classifier', type=str, help='LP | Classifier')
    parser.add_argument('--noisy_rate', default=0.4, type=float, help='labeling rate') # 0.6 noise rate
    
    
    config = parser.parse_args()
    
    config.data_dir = './data/' + config.dataset
    config.temp_save_path = './temp/' + config.dataset + '/'
    config.model_save_path = './checkpoints/' + config.dataset
    
    
    # config = vars(args)
    args = vars(config)
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ','')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]
    
    
    sys.stdout = Logger("result/"+ config.dataset +".log", sys.stdout)
    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')
        
    main(config)
    

