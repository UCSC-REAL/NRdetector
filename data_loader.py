
import os
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from sklearn.preprocessing import StandardScaler


class SegLoader(Dataset):
    def __init__(self, data_dir, split_size, split, **kwargs): 
        # split:train,val,test
        super().__init__()
        self.data_dir = data_dir
        # print(self.data_dir.split('/')[-1])

        self.split_size = split_size
        self._load_data(data_dir, split)

    def _load_data(self, data_dir, split):
        
        data, dlabel, wlabel = [], [], []
        data_ = np.load(os.path.join(data_dir, split)+ "/data.npy") 
        label_ = np.load(os.path.join(data_dir, split)+ "/label.npy")

        # data_, label_, (length, input_size) = np.load(filepath, allow_pickle=True)
        if self.data_dir.split('/')[-1] == 'PSM':
            label_ = label_.squeeze()
        data_, dlabel_, wlabel_ = self._preprocess(data_, label_, self.split_size)
            
        data.append(data_)
        dlabel.append(dlabel_)
        wlabel.append(wlabel_)
        # self.input_size = input_size
        self.data = torch.cat(data, dim=0)
        self.dlabel = torch.cat(dlabel, dim=0)
        self.wlabel = torch.cat(wlabel, dim=0) 
        
        # print(self.data.size())   
        # print(self.dlabel.size())
        # print(self.wlabel.size())    

                    
             
    def _preprocess(self, data, label, split_size):

        # normalize
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)

        # split
        data = torch.Tensor(data)
        data = f.pad(data, (0, 0, split_size - data.shape[0] % split_size, 0), 'constant', 0)
        data = torch.unsqueeze(data, dim=0)
        data = torch.cat(torch.split(data, split_size, dim=1), dim=0)

        label = torch.Tensor(label)
        label = f.pad(label, (split_size - label.shape[0] % split_size, 0), 'constant', 0)
        label = torch.unsqueeze(label, dim=0)
        label = torch.cat(torch.split(label, split_size, dim=1), dim=0)
        
        dlabel = label
        wlabel = torch.max(label, dim=1)[0]
        
        return data, dlabel, wlabel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'dlabel': self.dlabel[idx],
            'wlabel': self.wlabel[idx]
        } 
   
class PULoader(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        return sample, label



def get_segment(args):

  
    train_dataset = SegLoader(args.data_dir, args.win_size, 'train')
    valid_dataset = SegLoader(args.data_dir, args.win_size, 'valid')
    test_dataset = SegLoader(args.data_dir, args.win_size, 'test')
    # whole dataset
    combined_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=10000, shuffle=False, num_workers=1)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=10000, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False, num_workers=1)
    
    dataloader = DataLoader(dataset=combined_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    print("The information of the dataset:")
    # list = tensor.numpy().tolist()
    for _, train_batch in enumerate(train_loader):
        train_data = train_batch['data']
        train_wlabel = train_batch['wlabel']
        train_dlabel = train_batch['dlabel']
        print("The size of train_data:",train_data.size())
        print("The num of anomaly instances:",torch.sum(train_wlabel))
    for _, valid_batch in enumerate(valid_loader):
        valid_data = valid_batch['data']
        valid_wlabel = valid_batch['wlabel']
        valid_dlabel = valid_batch['dlabel']
        print("The size of valid_data:",valid_data.size())
        print("The num of anomaly instances:",torch.sum(valid_wlabel))
    for _, test_batch in enumerate(test_loader):
        test_data = test_batch['data']
        test_wlabel = test_batch['wlabel']
        test_dlabel = test_batch['dlabel']
        print("The size of test_data:",test_data.size())
        print("The num of anomaly instances:",torch.sum(test_wlabel))
        print("The num of anomaly points:",torch.sum(test_dlabel))
    data_ = torch.cat((train_data,valid_data,test_data),dim=0)
    wlabel_ = torch.cat((train_wlabel,valid_wlabel,test_wlabel),dim=0)
    dlabel_ = torch.cat((train_dlabel,valid_dlabel,test_dlabel),dim=0)
    print("The whole data size:",data_.size()) # data: torch.Size([1452, 100, 8])
 
    data = data_.tolist()
    wlabel = wlabel_.tolist()
    dlabel = dlabel_.tolist() 
    train_wlabel_ = train_wlabel.tolist()
    valid_wlabel_ = valid_wlabel.tolist()
    test_wlabel_ = test_wlabel.tolist()
    train_i = []
    train_n = []
    valid_i = []
    valid_n = []
    test_i = []
    test_n = []
    for i in range(len(train_wlabel_)):
        if train_wlabel_[i]>0:
            train_i.append(i+1)
        else:
            train_n.append(i+1)
    for i in range(len(valid_wlabel_)):
        if valid_wlabel_[i]>0:
            valid_i.append(i+1)
        else:
            valid_n.append(i+1)
    for i in range(len(test_wlabel_)):
        if test_wlabel_[i]>0:
            test_i.append(i+1)
        else:
            test_n.append(i+1)

    return train_i, train_n, valid_i, valid_n, test_i, test_n, data, wlabel, dataloader, dlabel


