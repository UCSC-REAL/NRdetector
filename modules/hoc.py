import argparse
import numpy as np
import pandas as pd
from os import path

import time
import torch, os
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from utils import *
from models.resnet import resnet18
    
    
    
from utils import *
import torch
smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)
    

class TrainSet(Dataset):
    def __init__(self, data, label, file_path=None):
        

        if file_path is not None:
            label = np.load("./data_from_pulp/hoc_pred.npy").flatten()
            data = np.load("./data_from_pulp/fake_instance.npy")
            
        
        self.data = torch.tensor(data[:,np.newaxis]) #一维的anomaly transformer
        self.label = label.astype(int)
        print("current noisy y=1:",sum(label)/len(label))
        
        print(self.data.size())
        print(label.shape)
            
    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], np.array(self.label[idx]), idx  
   
   
def count_y(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    max_val = np.max(dist)
    am = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist,axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt     


def func(KINDS, p_estimate, T_out, P_out, N,step, LOCAL, _device):
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)       # define the loss

    P = smp(P_out)
    T = smt(T_out)

    mode = random.randint(0, KINDS-1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N, N*N*N
    p_temp = count_real(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    weight = [1.0,1.0,1.0]
    # weight = [2.0,1.0,1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j]) #/ np.sqrt(N**j)

    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P+eps))/10

    return loss


def calc_func(KINDS, p_estimate, LOCAL, _device, max_step = 501, T0=None, p0 = None, lr = 0.1):
    # init
    # _device =  torch.device("cpu")
    N = KINDS
    eps = 1e-8
    if T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N,N))
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device = None) / N + torch.rand((N,1), device = None)*0.1     # P：0-9 distribution
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr = lr)

    # train
    loss_min = 100.0
    T_rec = torch.zeros_like(T)
    P_rec = torch.zeros_like(P)

    # time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(KINDS, p_estimate, T, P, N,step, LOCAL, _device)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        # if step % 100 == 0:
        #     print('loss {}'.format(loss))
        #     print(f'step: {step}  time_cost: {time.time() - time1}')
        #     print(f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
        #     print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
            # time1 = time.time()

    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def get_T_global_min(record, max_step = 501, T0 = None, p0 = None, lr = 0.1, NumTest = 50, all_point_cnt = 15000):
    total_len = sum([len(a) for a in record])
    # print(record[1][0]['feature'])
    origin_trans = torch.zeros(total_len, record[1][0]['feature'].shape[0])
    origin_label = torch.zeros(total_len).long()
    cnt, lb = 0, 0
    for item in record:
        for i in item:
            origin_trans[cnt] = i['feature']
            origin_label[cnt] = lb
            cnt += 1
        lb += 1
    data_set = {'feature': origin_trans, 'noisy_label': origin_label}
    print("==================================")
    # Build Feature Clusters --------------------------------------
    KINDS = 2 # num_classes
    # NumTest = 50
    # all_point_cnt = 15000


    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        # print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace= True)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    device = set_device()
    loss_min, E_calc, P_calc, T_init = calc_func(KINDS, p_estimate, False, device, max_step, T0, p0, lr = lr)

    E_calc = E_calc.cpu().numpy()
    T_init = T_init.cpu().numpy()
    return E_calc, T_init



def get_hoc_threshold(data, label):
    device = set_device()
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    print(sum(label), len(label))
    kk = sum(label)/len(label)
    
    num_classes = 2
    
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    
    train_dataset = TrainSet(data, label)
    train_dataloader_EF = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=32,
                                                    shuffle=True,
                                                    num_workers=1,
                                                    drop_last=False)
    
    model.eval()
    record = [[] for _ in range(num_classes)]

    for i_batch, (feature, label, index) in enumerate(train_dataloader_EF):

        extracted_feature = feature.to(device)
        label = label.to(device)
        for i in range(extracted_feature.shape[0]):
            record[int(label[i])].append({'feature': extracted_feature[i].unsqueeze(-1).detach().cpu(), 'index': int(index[i].detach().cpu())})
    # print(record)
    new_estimate_T, p = get_T_global_min(record, max_step=1500, lr = 0.1, NumTest = 50)
    print(f'Estimation finished!')
    np.set_printoptions(precision=1)
    print(f'The estimated T (*100) is \n{new_estimate_T*100}')
    # cal the actual y=1
    
    print("The threshold before estimation is:", kk)
    a = (kk- new_estimate_T[0][1])/(new_estimate_T[1][1]-new_estimate_T[0][1])
    print("The threshold is:", a)
    return a
    
    


