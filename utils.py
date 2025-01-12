import numpy as np
import torch, math
from sklearn import metrics
import matplotlib.pyplot as plt
import random, os
import networkx as nx
import pandas as pd
from collections import defaultdict

# For negative selector
from re import A
import numpy as np
from numpy import *
from numpy.matlib import repmat
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
import pandas as pd
import networkx as nx
# import torch
# import torch.nn.functional as f
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class MatricesCalc():
    def __init__(self, save_path, data, dataset_name, level="instance", type = "vector"):
        self.indexes = []
        for i in range(len(data)):
            self.indexes.append(i+1)
        self.data = data # [967,500,8]的一个列表
        self.arg_k = 5
        self.arg_a = 0.0005
        # save_path = save_path + dataset_name + "/"
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        #     print("Create the directory: %s" % save_path)
            
        if level == "instance":
            
            self.adj_matrix_file = save_path + dataset_name + "_adj_matrix.csv"
            self.matrix_w_output = save_path + dataset_name + "_w_matrix.csv"
            self.graphml_output_file = save_path + dataset_name + "_knn_5.graphml"
           
        elif level == "point":
            
            self.adj_matrix_file = "./out/point_adj_matrix.csv"
            self.matrix_w_output = "./out/point_w_matrix.csv"
            self.graphml_output_file = "./out/point_knn_5.graphml"
        
        self.adjacence_matrix = self.calc_MatrizAdj(type)
        print("Get adjacence matrix.")
        self.ident_matrix = self.calc_Indent()
        print("Get ident matrix")
        self.A = None
        self.G = None
        self.calc_knn_w_matrices()	
        
    def calc_Indent(self):
        
        ident_matrix = np.identity(len(self.adjacence_matrix), dtype = float)
        ident_matrix = pd.DataFrame(ident_matrix, columns=self.indexes, index=self.indexes)
        return ident_matrix

    def calc_MatrizAdj(self, mode = "origin"):
        if mode == "origin":
           
            x = np.array(self.data)
            # x = x.reshape(x.shape[0]*x.shape[1],8)
            x = x.reshape(x.shape[0],-1)
            print(x.shape)# (1452, 800)
            
            Y = cdist(x,x,metric=cosine)
            

        elif mode == "vector":
            
            h_scores = self.data
            Y = cdist(h_scores, h_scores, metric=cosine)
        
        elif mode == "score":
            
            x = self.data
            # Y = cdist(x,x,metric=cosine)
            Y = cosine(x,x)

        else:
            pass

        adjacence_matrix = pd.DataFrame(Y, columns=self.indexes, index=self.indexes)
        adjacence_matrix.to_csv(path_or_buf=self.adj_matrix_file)
        return adjacence_matrix

    def PU_LP_knn(self, k):
            print("====== Doing KNN to Get Graphml ======")
            total_columns = self.adjacence_matrix.shape[0]
            
            A = pd.DataFrame(0, columns=self.adjacence_matrix.index, index=self.adjacence_matrix.index)
     
            G = nx.Graph()
            for index_i, row in self.adjacence_matrix.iterrows():
            
                knn = [1000 for temp in range(k)]
              
                knn_names = ['' for temp in range(k)]
                max_value = 1000
                max_value_id = 0
               
                for name_j, value in row.iteritems():
                    #se i != j faça
                    if(index_i != name_j):
                        
                        if (value < max_value):
                           
                            knn_names[max_value_id] = name_j
                            knn[max_value_id] = value
                            max_value_id = np.argmax(knn)
                            max_value = knn[max_value_id]
                for j in range(k):
                   
                    vizinho = knn_names[j]
                    A.loc[index_i][vizinho] = 1
                   
                    G.add_edge(index_i, vizinho, weight=1)
            
            print("Get the graph")
            nx.write_graphml(G, self.graphml_output_file)
            return A, G

    def calc_knn_w_matrices(self):
        print("Calculating k-NN matrices", flush=True)
        self.A, self.G = self.PU_LP_knn(self.arg_k)
      
        A = self.A.mul(self.arg_a)
        I = self.ident_matrix.subtract(A)
            
        I = pd.DataFrame(np.linalg.pinv(I.values.astype(np.float32)), columns=self.indexes, index=self.indexes)
        W = I.subtract(self.ident_matrix)

        W.to_csv(path_or_buf=self.matrix_w_output)
        print('Save Matriz W', flush=True)
        


# For HOC
def set_device():
    
    if torch.cuda.is_available():
        _device = torch.device("cuda:1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device

def distCosine(x, y):
    """
    :param x: m x k array
    :param y: n x k array
    :return: m x n array
    """
    xx = np.sum(x ** 2, axis=1) ** 0.5
    x = x / xx[:, np.newaxis]
    yy = np.sum(y ** 2, axis=1) ** 0.5
    y = y / yy[:, np.newaxis]
    dist = 1 - np.dot(x, y.transpose())  # 1 - cosine distance
    return dist


def count_real(KINDS, T, P, mode, _device = 'cpu'):
    # time1 = time.time()
    P = P.reshape((KINDS, 1))
    p_real = [[] for _ in range(3)]

    p_real[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    # p_real[2] = torch.zeros((KINDS, KINDS, KINDS)).to(_device)
    p_real[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        p_real[1] = torch.cat([p_real[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with p_estimate
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                p_real[2][r][(i+r+KINDS)%KINDS] = temp33[r]
        else:
            p_real[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]


    temp = []       # adjust the order of the output (N*N), keeping consistent with p_estimate
    for p1 in range(KINDS):
        temp = torch.cat((p_real[1][p1, KINDS-p1:], p_real[1][p1, :KINDS-p1]))
        p_real[1][p1] = temp
    return p_real

# For evaluation
def compute_wacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_dacc(pred, label):
    correct = torch.mul(pred, label)
    TP, T, P = int(torch.sum(correct)), int(torch.sum(label)), int(torch.sum(pred))
    return np.array([TP, T, P])

def compute_auc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    return metrics.auc(fpr, tpr)

def compute_bestf1(score, label, return_threshold=False):
    if isinstance(score, torch.Tensor):
        score = score.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()

    indices = np.argsort(score)[::-1]
    sorted_score = score[indices]
    sorted_label = label[indices]
    true_indices = np.where(sorted_label == 1)[0]

    bestf1 = 0.0
    best_threshold=None
    T = sum(label)
    for _TP, _P in enumerate(true_indices):
        TP, P = _TP + 1, _P + 1
        precision = TP / P
        recall = TP / T
        f1 = 2 * (precision*recall)/(precision+recall)
        threshold = sorted_score[_P] - np.finfo(float).eps
        if f1 > bestf1: # and threshold <= 0.5:
            bestf1 = f1
            best_threshold = sorted_score[_P] - np.finfo(float).eps
            #best_threshold = (sorted_score[_P-1] + sorted_score[_P]) / 2
    if return_threshold:
        return bestf1, best_threshold
    else:
        return bestf1

def compute_auprc(pred, label):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy().flatten()
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().flatten()
    return metrics.average_precision_score(label, pred)

def compute_precision_recall(result):
    TP, T, P = result
    recall, precision, IoU = 1, 1, TP / (T + P - TP)  
    if T != 0: recall = TP / T
    if P != 0: precision = TP / P

    if recall==0.0 or precision==0.0:
        f1 = 0.0
    else:
        f1 = 2*(recall*precision)/(recall+precision)
    return precision, recall, f1, IoU