import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import get_segment, PULoader
from modules.extractor import get_h_scores
from modules.selector import PU_LP
from modules.models import SimpleClassifier, create_loss, Classifier_six_layer, constraint_loss
from modules.hoc import get_hoc_threshold
from utils import MatricesCalc
from evaluation import evaluation, get_all_res
SIM_READY = True
DEBUG = True
# DEBUG = False

class Solver(object):

    def __init__(self, config):
        self.args = config
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        
        self.model_save_path = config.model_save_path
        self.temp_save_path = config.temp_save_path  # temp/EMG/
        
        if not os.path.exists(self.temp_save_path):
            os.mkdir(self.temp_save_path)
            print("Create the directory: %s" % self.temp_save_path)
        
        self.graphml_path = self.temp_save_path + self.dataset + "_knn_5.graphml"
        self.matrix_path = self.temp_save_path + self.dataset + "_w_matrix.csv"
        self.k = self.args.anomaly_ratio
 
    def begin(self):
        # Loading the data
        train_i, train_n, valid_i, valid_n, test_i, test_n, data, wlabel, dataloader, dlabel = get_segment(self.args)
        # Loading the pretrained encoding model
        scores, d_score, represents, dscores = get_h_scores(self.args, dataloader) 
        
        # ======== Negative Selector ========== #
        # Cal the similarity matrices and form the graph
        if not SIM_READY:
            print("Cal the similarity matrices and form the graph.")
            M = MatricesCalc(self.temp_save_path, scores, self.args.dataset)
        
        # pu-lp
        print("Start to PU-LP.")
        RP_path = self.temp_save_path + self.dataset + "_RP.npy"
        RN_path = self.temp_save_path + self.dataset + "_RN.npy"
        selector = PU_LP(self.temp_save_path, self.graphml_path, self.matrix_path)
        selector.begin(self.args.noisy_rate, train_i, train_n, valid_i, valid_n, test_i, test_n, wlabel)
        
        if not DEBUG:
            RP,RN = selector.train_()
            np.save(RP_path,RP)
            np.save(RN_path,RN)
            print("Get the P and filtered U")
        else:
            RP= np.load(RP_path)
            RN = np.load(RN_path)
            
        print("Finish select the P and U.")
        self.RP = RP
        self.RN = RN
        self.weak_scores = scores
        # self.weak_scores = dscores
        self.dense_scores = dscores # Actually is the act map
        self.represents = represents 
        self.s_index = len(train_i)+len(train_n)+len(valid_i)+len(valid_n)
        self.wlabel = wlabel
        self.dlabel = dlabel
        
        self.num_classes = 1
        self.model = self.build_model()
        
    def get_dataset(self):
        positive = []
        negative = []
        data_feature = self.weak_scores
        for item in self.RP:
            positive.append(data_feature[int(item)-1])
        for i in self.RN:
            negative.append(data_feature[int(i)-1])
            
        positive_samples = torch.tensor(positive, dtype=torch.float32)
        negative_samples = torch.tensor(negative, dtype=torch.float32)
        print("positive_samples:", positive_samples.shape, negative_samples.shape)
        if self.num_classes == 1:
            positive_labels = torch.tensor([1.0 for _ in range(len(positive))], dtype=torch.float32)
            negative_labels = torch.tensor([0.0 for _ in range(len(negative))], dtype=torch.float32)
        else:
            positive_labels = torch.tensor([[1.0, 0.0] for _ in range(len(positive))], dtype=torch.float32)
            negative_labels = torch.tensor([[0.0, 1.0] for _ in range(len(negative))], dtype=torch.float32)
        inputs = torch.cat((positive_samples, negative_samples), dim=0)
        labels = torch.cat((positive_labels, negative_labels), dim=0)
        
        dataset = PULoader(inputs, labels)
        batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        return dataloader
       
    def build_model(self):
        # self.model = SimpleClassifier(self.args.d_model, hidden_size=self.args.hidden_size, num_classes=self.num_classes).to(self.device)
        self.model = Classifier_six_layer(self.args.d_model, hidden_size=128, num_classes=self.num_classes).to(self.device)
        return self.model
     
    def train(self):
        dataloader = self.get_dataset()

      
        criterion = create_loss(self.args.prior, self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        

        # 训练模型
        num_epochs = self.args.n_epochs
        best_loss = 0.0
        
        for epoch in range(num_epochs):
        
            optimizer.zero_grad()
            self.model.train() 
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, out = self.model(inputs)
               
                lamda = 1
                
                lamda_1 = 1
                pu_loss = criterion(outputs, labels.unsqueeze(-1))
                const_loss = constraint_loss(out, labels, self.batch_size)
                loss = lamda_1 * pu_loss + lamda * const_loss

                loss.backward()
                optimizer.step()
                
            if epoch == 0:
                best_loss = loss
        
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                print(f'pu_loss: {pu_loss.item():.4f}, constraint_loss: {const_loss.item():.4f}')
            
            if loss < best_loss:

                
                torch.save(self.model.state_dict(), self.model_save_path + '_best_model.pth')
                best_loss = loss
                   
    def test(self):
        
        test_data = torch.tensor(self.weak_scores[self.s_index:], dtype=torch.float32)
        test_label = self.wlabel[self.s_index:]
        self.model.load_state_dict(torch.load(self.model_save_path + '_best_model.pth'))
        self.model.eval()
        with torch.no_grad():
            
            test_outputs, out = self.model(test_data.to(self.device))
            if self.num_classes == 1:
                anomaly_ratio = self.args.anomaly_thre
                threshold = torch.mean(test_outputs)+ anomaly_ratio * (torch.max(test_outputs)-torch.min(test_outputs))
               
                predicted_labels = (test_outputs >= threshold)
                
                pred_label = predicted_labels.cpu().numpy().astype(int)
                
            else:
                predicted_labels = (test_outputs >= 0.5)
                
                pred_label = []
                for item in predicted_labels:
                    if item[0]>item[1]:
                        pred_label.append(1)
                    else:
                        pred_label.append(0)
    
        
        evaluation(np.array(pred_label), np.array(test_label))
        self.instance_label = pred_label
        self.gt, self.interested_instance, self.interested_index = self.save_instance_files(pred_label)
        print("Save the instance files...")
        print("Finish testing.")
        return pred_label
    
    def save_instance_files(self, pred_label):
        gt = np.array(self.dlabel[self.s_index:]).astype(int)
        test_score = np.array(self.dense_scores[self.s_index:])
        test_logit = np.array(self.represents[self.s_index:])
        interested_instance = []
        interested_represent = [] 
        i_index = []
        for i in range(len(self.instance_label)):
            if self.instance_label[i] > 0:
                # 是异常instance
                interested_instance.append(test_score[i])
                interested_represent.append(test_logit[i])
                i_index.append(i+1)
        
        np.save(self.temp_save_path + "test_labels.npy", gt)   
        np.save(self.temp_save_path + "test_scores.npy",test_score)     
        np.save(self.temp_save_path + "interested_instance.npy",np.array(interested_instance))
        np.save(self.temp_save_path + "interested_represent.npy",np.array(interested_represent))
        np.save(self.temp_save_path + "interested_index.npy", np.array(i_index))
        return gt, np.array(interested_instance), np.array(i_index)
        
    def rank_test(self, debug = True):
        if debug:
            self.interested_index = np.load(self.temp_save_path +"interested_index.npy")
            self.interested_instance = np.load(self.temp_save_path +"interested_instance.npy", allow_pickle=True)
            self.gt = np.load(self.temp_save_path +"test_labels.npy") # 7456
        
        pred = np.zeros_like(self.gt)
        point_Score = self.interested_instance.reshape(-1)
        point_index = np.argsort(point_Score) 
        
        print("interested_instance", self.interested_instance.shape) # (351, 100)
        print("point score:", point_Score.shape) 
        print("test_labels:", self.gt.shape) #(1309, 100)
        
        
        hoc_gt = []
        hoc_pred = []
        hoc_score = []
        
        
        sorted_num = self.k*len(point_Score)
        for i in range(len(point_Score)):
            
            cur_index = int(point_index[len(point_Score)-i-1])
            row = self.interested_index[int(cur_index / 100)]-1 # 1-based
            col = int(cur_index % 100)
            if i < sorted_num:
                
                pred[row][col] = 1
            hoc_gt.append(self.gt[row][col])
            hoc_pred.append(pred[row][col])
            hoc_score.append(point_Score[cur_index])
           
        # dscores = np.load(self.temp_save_path + "test_scores.npy")
        # pred_ = time_based_lp_sim(pred, dscores, k=1)
        gt = self.gt.reshape(-1).astype(int)     
        pred = pred.reshape(-1).astype(int)
        
        
        # 增加hoc估计器
        print("Save the hoc files...")
        print(np.array(hoc_gt).shape, np.array(hoc_pred).shape, np.array(hoc_score).shape)  
        np.save(self.temp_save_path + "hoc_gt.npy",np.array(hoc_gt))
        np.save(self.temp_save_path + "hoc_pred.npy",np.array(hoc_pred))
        np.save(self.temp_save_path + "hoc_score.npy",np.array(hoc_score))
        
        a = get_hoc_threshold(np.array(hoc_score), np.array(hoc_pred))
        self.rank_hoc(a)

        get_all_res(pred, gt)
            
    def pick_test(self, debug = True):
        if debug:
            self.interested_index = np.load(self.temp_save_path +"interested_index.npy")
            self.interested_instance = np.load(self.temp_save_path +"interested_instance.npy", allow_pickle=True)
            self.gt = np.load(self.temp_save_path +"test_labels.npy") # 7456
        point_Score = self.interested_instance.reshape(-1)
        
        k = 0.3
        pred = np.zeros_like(self.gt)
        sorted_num = k * self.args.win_size
        point_DP = []
        point_DU = [n for n in range(1, len(point_Score)+1)]
        count = 0
        for instance in self.interested_instance:
            # sorted_index = np.where(instance == np.max(instance))[0]
            sorted_index = np.argsort(instance)
            for i in range(len(sorted_index)):

                if i < sorted_num:
                    max_index =sorted_index[len(sorted_index) -1-i] 
                    point_index = self.args.win_size*count + max_index+1 
                    point_DP.append(point_index)
                    point_DU.remove(point_index)
            count += 1

        
        if DEBUG:
            print("Ground Truth:", self.gt.shape)
            print("interested_instance", self.interested_instance.shape)
            print(point_Score.shape)
            print("sorted_num,",sorted_num)

        P = point_DP[:]
        U = point_DU[:]

        lp_i = []
        point_label = []
        for i in range(len(point_Score)):
            cur_p = i + 1 # 1-based point index.
            if cur_p in point_DP:
                point_label.append(1)
            elif cur_p in lp_i:
                point_label.append(1)
            else:
                point_label.append(0)
                
        print("label:",len(point_label))
        label = np.array(point_label).reshape(-1,self.args.win_size) # lp_i+P, 100,8
        row = 0
        for idx in self.interested_index:
            # 1-based
            pred[idx-1] = label[row]
            row += 1
            
        np.save("label.npy",np.array(pred))
        # np.save("data.npy",np.array(point_Score))
        
        print("========Get all labels of the test data.==========")
        gt = self.gt.reshape(-1).astype(int)       
        pred = pred.reshape(-1).astype(int)

        evaluation(pred, gt)
        # get_all_res(pred, gt)

    def rank_hoc(self,k, debug = True):
        pred = np.zeros_like(self.gt)
        point_Score = self.interested_instance.reshape(-1)
        point_index = np.argsort(point_Score) 
        
        print("Current k is:", k)
        sorted_num = k * len(point_Score)
        for i in range(len(point_Score)):
          
            cur_index = int(point_index[len(point_Score)-i-1])
            row = self.interested_index[int(cur_index / 100)]-1 # 1-based
            col = int(cur_index % 100)
            if i < sorted_num:
               
                pred[row][col] = 1
    
           
        gt = self.gt.reshape(-1).astype(int)     
        pred = pred.reshape(-1).astype(int)
        get_all_res(pred, gt)
        

def time_based_label_propagation(results,k=1):
    pred = []
    for item in results:
        pred.append(item)
    for i in range(1,len(pred)-1):
       
        if results[i] == 1:
            for kk in range(1,k+1):
                pred[i-kk] = 1
            
                pred[i+kk] = 1
    return pred


def time_based_lp_sim(results,scores,k=1):

    pred = []
    for item in results:
        pred.append(item)
    anomaly_ratio = 0.3 
    for j in range(len(pred)):
        thre = (max(scores[j])-min(scores[j])) * anomaly_ratio + min(scores[j])
        for i in range(1,len(pred[j])-1):
    
            if results[j][i] == 1:
                if abs(scores[j][i] - scores[j][i-1]) < thre:
                    for kk in range(1,k+1):
                        pred[j][i-kk] = 1
                        
                if abs(scores[j][i]-scores[j][i+1]) < thre:
                    for kk in range(1,k+1):
                        pred[j][i+kk] = 1
    return np.array(pred).reshape(-1)