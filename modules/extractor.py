import os, time
import argparse
import numpy as np
from modules.softdtw_cuda import SoftDTW
from utils import *
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.transformer import AnomalyTransformer




def get_h_scores(args, data_loader):
    
    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)

    # Random seed initialization
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import random

    random.seed(args.seed)

    args.data_dir = './data/' + args.dataset

    # valid_loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # Select Device
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    
    dtw = SoftDTW(use_cuda=True, gamma=args.gamma, normalize=False)
    model = DilatedCNN(input_size=args.data_feature,
                       hidden_size=args.hidden_size,
                       output_size=args.output_size,
                       kernel_size=args.kernel_size,
                       n_layers=args.n_layers,
                       pooling_type=args.pooling_type,
                       local_threshold=args.local_threshold,
                       granularity=args.granularity,
                       beta=args.beta,
                       split_size=args.win_size,
                       dtw=dtw)
    # model = AnomalyTransformer(dtw = dtw, pooling_type=args.pooling_type, win_size=args.win_size, enc_in=args.data_feature, c_out=args.output_size, e_layers=7)
    
    model.cuda()
    # model.to(device)
    
    if args.win_size == 100:
        # model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_MODEL.pth")) # 0.8 cnn
        # model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_model.pth")) # 0.8 transformer
        # model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_model_2.pth")) # 0.2 transformer 3
        # model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_model_4_7.pth")) # 0.2 transformer 7
        model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_model_4.pth")) # 0.4 cnn
        # model.load_state_dict(torch.load("./pretrained_model/model.pth")) # 0.4 cnn smd
        # model.load_state_dict(torch.load("./pretrained_model/"+args.dataset+"_model_6.pth")) # 0.6 cnn 
    
    bce = nn.BCELoss(reduction='mean')
    wscores, dscores, output, actmap = evaluation(args, model, data_loader, bce,100)

    w_score = []
    for score in wscores:
        score_ = score.tolist()
        for item in score_:
            w_score.append(item)
    print(len(w_score), len(w_score[0])) 
    
    d_score = []
    for score in dscores:
        score_ = score.tolist()
        for item in score_:
            d_score.append(item)
    print(len(d_score), len(d_score[0])) # 4353 100
    
    represent = []
    for output in output:
        output_ = output.tolist()
        for item in output_:
            represent.append(item)
    print(len(represent), len(represent[0]),len(represent[0][0]))  
    
    actmap_ = []
    for item in actmap:
        item_ = item.tolist()
        for i in item_:
            actmap_.append(i)
    print(len(actmap_), len(actmap_[0]))
    
    # 返回w_score
    return w_score, d_score, represent, actmap_
    

def evaluation(args, model, data_loader, bce, epoch):
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    model.eval()
    total, total_loss, total_bce_loss, total_dtw_loss = 0, 0, 0, 0
    wscores, wlabels = [], []
    dscores, dlabels = [], []
    wresult, dresult = np.zeros(3), np.zeros(3)
    represents = []
    actmap = []
    # for itr, batch in enumerate(data_loader):
    for batch in data_loader:
        
        data = batch['data'].cuda()
        wlabel = batch['wlabel'].cuda()
        dlabel = batch['dlabel'].cuda()
        
        # data = batch['data'].to(device)
        # wlabel = batch['wlabel'].to(device)
        # dlabel = batch['dlabel'].to(device)
        
        with torch.no_grad(): 
            out = model.get_scores(data)
            # print(out['wscore'])
            bce_loss = bce(out['wscore'], wlabel)
            dtw_loss = model.dtw_loss(out['output'], wlabel).mean(0)
            loss = bce_loss + dtw_loss

            total += data.size(0)
            total_bce_loss += bce_loss.item() * data.size(0)
            total_dtw_loss += dtw_loss.item() * data.size(0)
            total_loss += loss.item() * data.size(0)
            # print("out['output']",out['output'].size())
            represents.append(out['output']) 
            # represents.append(out['p']) 
            # weak prediction
            wresult += compute_wacc(out['wpred'], wlabel)
            # wscores.append(out['wscore'])
            # wscores.append(out['output'])
            wscores.append(out['h'])
            wlabels.append(wlabel)

            # dense prediction
            dpred,actm = model.get_dpred(out['output'], out['wpred'])
            dresult += compute_dacc(dpred, dlabel)
            dscores.append(out['dscore'])
            dlabels.append(dlabel)
            actmap.append(actm)
    return wscores, dscores, represents,actmap
 

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super(ResidualBlock, self).__init__()

        self.dilation = dilation

        self.diconv = nn.Conv1d(in_channels=input_size,
                                out_channels=output_size,
                                kernel_size=kernel_size,
                                dilation=dilation)

        self.conv1by1_skip = nn.Conv1d(in_channels=output_size,
                                       out_channels=output_size,
                                       kernel_size=1,
                                       dilation=1)

        self.conv1by1_out = nn.Conv1d(in_channels=output_size,
                                      out_channels=output_size,
                                      kernel_size=1,
                                      dilation=1)

    def forward(self, x):
        x = f.pad(x, (self.dilation, 0), "constant", 0)
        z = self.diconv(x)
        z = torch.tanh(z) * torch.sigmoid(z)
        s = self.conv1by1_skip(z)
        z = self.conv1by1_out(z) + x[:,:,-z.shape[2]:]
        return z, s


class DilatedCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, n_layers, pooling_type='avg', granularity=1, \
      local_threshold=0.5, global_threshold=0.5, beta=10, split_size=500, dtw=None):
        super(DilatedCNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        
        self.rf_size = self.kernel_size ** (self.n_layers-1) # the size of receptive field
        self.pooling_type = pooling_type

        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.granularity = granularity
        self.beta = beta
        
        self.split_size = split_size
        self.dtw = dtw
        
        self.build_model(input_size, hidden_size, output_size, kernel_size, n_layers)

    def build_model(self, input_size, hidden_size, output_size, kernel_size, n_layers):
        # causal conv. layer
        self.causal_conv = nn.Conv1d(in_channels=input_size,
                                     out_channels=hidden_size,
                                     kernel_size=kernel_size,
                                     stride=1, dilation=1)

        # dilated conv. layer
        self.diconv_layers = nn.ModuleList()
        for i in range(n_layers):
            diconv = ResidualBlock(input_size=hidden_size,
                                   output_size=hidden_size,
                                   kernel_size=kernel_size,
                                   dilation=kernel_size**i)
            self.diconv_layers.append(diconv)

        # 1x1 conv. layer (for skip-connection)
        self.conv1by1_skip1 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=hidden_size,
                                        kernel_size=1, dilation=1)

        self.conv1by1_skip2 = nn.Conv1d(in_channels=hidden_size,
                                        out_channels=output_size,
                                        kernel_size=1, dilation=1)

        self.fc = nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    init.normal_(m.bias.data)

    def forward(self, x):
        x = x.transpose(1, 2)

        padding_size = self.rf_size - x.shape[2]
        if padding_size > 0:
            x = f.pad(x, (padding_size, 0), "constant", 0)
        
        x = f.pad(x, (1, 0), "constant", 0)
        z = self.causal_conv(x)

        out = torch.zeros(z.shape).cuda()
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        # out = torch.zeros(z.shape).to(device)
        # import pdb
        # pdb.set_trace()

        for diconv in self.diconv_layers:
            z, s = diconv(z)
            out += s

        out = f.relu(out)
        # out_plus = out.transpose(1,2)
        out = self.conv1by1_skip1(out)
        # out_plus = out.transpose(1,2)
        out = f.relu(out)
        out_plus = out.transpose(1,2)
        out = self.conv1by1_skip2(out).transpose(1, 2)
        # print("yes:",out.size()) # 32,100,64
        return out, out_plus

    def get_scores(self, x):
        ret = {}
        out, out_plus = self.forward(x)
        ret['output'] = out
        # print("out:",out.shape)
        ret['p'] = out_plus
        # Compute weak scores
        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1) 
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]
        ret['h'] = _out
        # print("out", _out.shape) # 32,64
        ret['wscore'] = torch.sigmoid(self.fc(_out).squeeze(dim=1))
        # print(torch.sigmoid(self.fc(_out).squeeze(dim=1)).shape)
        #print("wscore",ret['wscore'].shape)
        ret['wpred'] = (ret['wscore'] >= self.global_threshold).type(torch.cuda.FloatTensor)

        # Compute dense scores
        h = self.fc(out).squeeze(dim=2)
        # print(h.shape)
        ret['dscore'] = torch.sigmoid(h)
        # print(torch.sigmoid(h).shape) # 32,100
        ret['dpred'] = (ret['dscore'] >= self.local_threshold).type(torch.cuda.FloatTensor)
        return ret

    def get_seqlabel(self, actmap, wlabel):
        actmap *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1])
        seqlabel = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor)
        seqlabel = f.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity)))
        seqlabel = torch.max(seqlabel, dim=2)[0]
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1)
        # seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).to(device), seqlabel, torch.zeros(seqlabel.shape[0], 1).to(device)], dim=1)

        return seqlabel

    def dtw_loss(self, out, wlabel):
        h = self.fc(out).squeeze(dim=2) 
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2))

        with torch.no_grad():
            # Activation map
            actmap = h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            pos_seqlabel = self.get_seqlabel(actmap, wlabel)
            neg_seqlabel = self.get_seqlabel(actmap, 1-wlabel)

        pos_dist = self.dtw(pos_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        neg_dist = self.dtw(neg_seqlabel.unsqueeze(dim=1), dscore.unsqueeze(dim=1)) / self.split_size
        loss = f.relu(self.beta + pos_dist - neg_dist)
        return loss

    def get_alignment(self, label, score):
        # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)

    def get_dpred(self, out, wlabel):
        h = self.fc(out).squeeze(dim=2)
        dscore = torch.sigmoid(self.fc(out).squeeze(dim=2))

        with torch.no_grad():
            # Activation map
            actmap = h
            actmin = torch.min(actmap, dim=1)[0]
            actmap = actmap - actmin.unsqueeze(dim=1)
            actmax = torch.max(actmap, dim=1)[0]
            actmap = actmap / actmax.unsqueeze(dim=1)
            # Sequential labels
            seqlabel = self.get_seqlabel(actmap, wlabel)

        return self.get_alignment(seqlabel, dscore),actmap



