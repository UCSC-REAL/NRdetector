import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CLASS_PRIOR = {
    # 'EMG': 0.074,
    'EMG': 0.4,
    'SMD': 0.31,
    'MSL': 0.20,
    'SMAP': 0.174,
    'PSM':0.31
}




class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size,64),
            # nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            # nn.ReLU(),
            # nn.Linear(num_classes,1),
            nn.Sigmoid()
        )

        
    def forward(self, x):
        x = self.net(x)
        return x, x


class Classifier_six_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier_six_layer, self).__init__()
        self.pooling_type = 'avg'
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 100),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(100, 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        dscores = self.net(x) # batch, 100
        # print(dscores.size())
        # if self.pooling_type == 'avg':
        #     _out = torch.mean(dscores, dim=1) 
        # elif self.pooling_type == 'max':
        #     _out = torch.max(dscores, dim=1)[0]
       
        x = self.fc1(dscores)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # print(x.size(), _out.size()) # batch, 1, batch, 1
        return x, dscores


class LabelDistributionLoss(torch.nn.Module):
    def __init__(self, prior, device, num_bins=1, proxy='polar', dist='L1'):
        super(LabelDistributionLoss, self).__init__()
        self.prior = prior
        self.frac_prior = 1.0 / (2*prior)

        self.step = 1 / num_bins # bin width. predicted scores in [0, 1]. 
        self.device = device
        self.t = torch.arange(0, 1+self.step, self.step).view(1, -1).requires_grad_(False) # [0, 1+bin width)
        self.t_size = num_bins + 1

        self.dist = None
        if dist == 'L1':
            self.dist = F.l1_loss
        else:
            raise NotImplementedError("The distance: {} is not defined!".format(dist))

        # proxy
        proxy_p, proxy_n = None, None
        if proxy == 'polar':
            proxy_p = np.zeros(self.t_size, dtype=float)
            proxy_n = np.zeros_like(proxy_p)
            proxy_p[-1] = 1
            proxy_n[0] = 1
        else:
            raise NotImplementedError("The proxy: {} is not defined!".format(proxy))
        
        proxy_mix = prior*proxy_p + (1-prior)*proxy_n
        # print('#### Label Distribution Loss ####')
        # print('ground truth P:')
        # print(proxy_p)
        # print('ground truth U:')
        # print(proxy_mix)

        # to torch tensor
        self.proxy_p = torch.from_numpy(proxy_p).requires_grad_(False).float()
        self.proxy_mix = torch.from_numpy(proxy_mix).requires_grad_(False).float()

        # to device
        self.t = self.t.to(self.device)
        self.proxy_p = self.proxy_p.to(self.device)
        self.proxy_mix = self.proxy_mix.to(self.device)

    def histogram(self, scores):
        scores_rep = scores.repeat(1, self.t_size)
        
        hist = torch.abs(scores_rep - self.t)

        inds = (hist>self.step)
        hist = self.step-hist # switch values
        hist[inds] = 0

        return hist.sum(dim=0)/(len(scores)*self.step)

    def forward(self, outputs, labels):
        scores=torch.sigmoid(outputs)
        labels=labels.view(-1,1)

        scores = scores.view_as(labels)
        s_p = scores[labels==1].view(-1,1)
        s_u = scores[labels==0].view(-1,1)

        l_p = 0
        l_u = 0
        if s_p.numel() > 0:
            hist_p = self.histogram(s_p)
            l_p = self.dist(hist_p, self.proxy_p, reduction='mean')
        if s_u.numel() > 0:
            hist_u = self.histogram(s_u)
            l_u = self.dist(hist_u, self.proxy_mix, reduction='mean')

        return l_p + self.frac_prior * l_u


def create_loss(dataset_prior,device):
    prior = dataset_prior
    # print('prior: {}'.format(prior))
    base_loss = LabelDistributionLoss(prior=prior, device=device)
    return base_loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2-arr)**2)
    return lamda1*loss


def sparsity(arr, lamda2):
    loss = torch.sum(arr)
    return lamda2*loss


def ranking(scores, batch_size):
    loss = torch.tensor(0., requires_grad=True)
    for i in range(batch_size):
        maxn = torch.max(scores[int(i*32):int((i+1)*32)])
        maxa = torch.max(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)])
        tmp = F.relu(1.-maxa+maxn)
        loss = loss + tmp
        loss = loss + smooth(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)],8e-5)
        loss = loss + sparsity(scores[int(i*32+batch_size*32):int((i+1)*32+batch_size*32)], 8e-5)
    return loss / batch_size


def constraint_loss(dscores, wlabel, batch_size):
    loss = torch.tensor(0., requires_grad=True)
    a_instance = []
    n_instance = []
    for i in range(batch_size):
        if wlabel[i] > 0:
            # anomaly
            a_instance.append(dscores[i])
            # L2 = torch.mean(dscores[i])
        else:
            n_instance.append(dscores[i])
    
    if a_instance and n_instance:
        L2 = loss
        L1 = loss
        for anomaly in a_instance:
            L2 = L2 +  max(torch.mean(torch.stack(n_instance)) - torch.mean(anomaly), 0.0)
            L2 = L2 + max(smooth(anomaly, 8e-5),0.0) ## 
        for normal in n_instance:
            L1 = L1 + max(smooth(normal, 8e-5),0.0)
            
        return (L1 + L2) / batch_size
    else:
        if a_instance is None:
            L1 = loss
            L1 = L1 + max(smooth(torch.stack(n_instance), 8e-5),0.0)
            return L1 / batch_size
        elif n_instance is None:
            L3 = loss
            L3 = L3 + max(torch.max(torch.stack(a_instance)) - torch.mean(torch.stack(a_instance)), 0.0)
            L3 = L3 + max(smooth(torch.stack(a_instance), 8e-5),0.0) # 
            
            return L3 / batch_size
        
    return loss / batch_size