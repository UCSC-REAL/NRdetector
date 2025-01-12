import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding

# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn= self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # new_x, attn, mask, sigma = self.attention(
        #     x, x, x,
        #     attn_mask=attn_mask
        # )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series= attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            # prior_list.append(prior)
            # sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list


class AnomalyTransformer(nn.Module):
    def __init__(self, dtw, pooling_type,  win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=False):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.pooling_type = pooling_type
        self.local_threshold = 0.5
        self.global_threshold = 0.5
        self.granularity = 4
        self.beta = 0.4
        self.dtw = dtw
        self.split_size = win_size
        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.fc = nn.Linear(c_out, 1)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        # print(enc_out.shape) # 64,100,512
        enc_out, series = self.encoder(enc_out)
        enc_out = self.projection(enc_out)
        # print(enc_out.shape) # 64,100,64(输出特征)
        if self.output_attention:
            return enc_out, series
        else:
            return enc_out, None
        
        
    def get_scores(self, x):
        ret = {}
        out, _ = self.forward(x)
        ret['output'] = out

        # Compute weak scores
        if self.pooling_type == 'avg':
            _out = torch.mean(out, dim=1) 
        elif self.pooling_type == 'max':
            _out = torch.max(out, dim=1)[0]
        ret['h'] = _out
        # print("_out:", _out.shape) # 64，64
        ret['wscore'] = torch.sigmoid(self.fc(_out).squeeze(dim=1))
        ret['wpred'] = (ret['wscore'] >= self.global_threshold).type(torch.cuda.FloatTensor)

        # Compute dense scores
        h = self.fc(out).squeeze(dim=2)
        ret['dscore'] = torch.sigmoid(h)
        ret['dpred'] = (ret['dscore'] >= self.local_threshold).type(torch.cuda.FloatTensor)
        return ret
    
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
        loss = F.relu(self.beta + pos_dist - neg_dist)
        return loss
        
    def get_seqlabel(self, actmap, wlabel):
        actmap *= wlabel.unsqueeze(dim=1).repeat(1, actmap.shape[1])
        seqlabel = (actmap >= self.local_threshold).type(torch.cuda.FloatTensor)
        seqlabel = F.pad(seqlabel, (self.granularity - seqlabel.shape[1] % self.granularity, 0), 'constant', 0)
        seqlabel = torch.reshape(seqlabel, (seqlabel.shape[0], -1, int(seqlabel.shape[1] / self.granularity)))
        seqlabel = torch.max(seqlabel, dim=2)[0]

        seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).cuda(), seqlabel, torch.zeros(seqlabel.shape[0], 1).cuda()], dim=1)
        # seqlabel = torch.cat([torch.zeros(seqlabel.shape[0], 1).to(device), seqlabel, torch.zeros(seqlabel.shape[0], 1).to(device)], dim=1)

        return seqlabel
    
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
    
    def get_alignment(self, label, score):
        # label : batch x pseudo-label length (= B x L)
        # score : batch x time-sereis length (= B x T)
        assert label.shape[0] == score.shape[0]
        assert len(label.shape) == 2  
        assert len(score.shape) == 2

        A = self.dtw.align(label.unsqueeze(dim=1), score.unsqueeze(dim=1))
        indices = torch.max(A, dim=1)[1]
        return torch.gather(label, 1, indices)

