import pandas as pd
import networkx as nx
import numpy as np
from evaluation import evaluation


class PU_LP():
    def __init__(self, temp_path, graphml_input_file, matrix_w_input_name):
        self.G = nx.read_graphml(graphml_input_file)            
        self.W = pd.read_csv(matrix_w_input_name, index_col=0)  
      
        self.output_file = temp_path + "result.csv"              
        self.class_info_file = temp_path +  "class_info.txt"      
        self.temp_path = temp_path
       
        self.m = 4
        self.lmbda = 0.32
        
       
        self.train = None
        self.test = None
        self.labels = None
        self.train_index = None
        self.test_index = None
        self.data_len = 0
        
    def begin(self, noisy_rate, train_i_index,train_n_index,valid_i_index,valid_n_index, test_i_index,test_n_index,label):

        self.train_DP = []
        self.train_DU = [] 
        train_known_num = int(noisy_rate*len(train_i_index))
        valid_known_num = int(noisy_rate*len(valid_i_index))
 
        for i in range(len(train_i_index)):
            if i < train_known_num:
                self.train_DP.append(train_i_index[i])
            else:
                self.train_DU.append(train_i_index[i])
        for item in train_n_index:
            self.train_DU.append(item)
        
        train_len = len(train_i_index) + len(train_n_index)
        for i in range(len(valid_i_index)):
            if i < valid_known_num:
                self.train_DP.append(valid_i_index[i]+train_len)
            else:
                self.train_DU.append(valid_i_index[i]+train_len)
        for item in valid_n_index:
            self.train_DU.append(item+train_len)
            
        self.test_DU = []  
        test_base = len(valid_i_index) + len(valid_n_index) + train_len
        for i in test_i_index:
            self.test_DU.append(i+test_base)
        for j in test_n_index:
            self.test_DU.append(j+test_base)
        self.labels = label
        self.data_len = 4353

        print("=========== Train PU-LP Start. =============")
        
    def train_(self):
        P = self.train_DP[:]     # D+
        U = self.train_DU[:]      # Du
        whole_U = U[:]
        
        P, RP, RN = self.calc_pu(P,  U) 
        results, labels, lp_i, lp_n = self.calc_lp(P, RN, whole_U, "train") 
        print("RP,RN train:",len(P)," ",len(RP)," ",len(RN)) # RP,RN train: 200   120   320
        print("len lp_i, lp_n:", len(lp_i), len(lp_n))
        print("len results:", len(results), len(labels))
        evaluation(results, labels)

        self.test_DP = P
        return P, lp_n

        
    def test_(self):
        
        # self.calc_lp()
        P = self.test_DP[:]
        U = self.test_DU[:]
        whole_test_U = U[:]
        P, RP, RN = self.calc_pu(P, U)
        print("RP,RN test:",len(P)," ",len(RP)," ",len(RN))
        results, labels, lp_i, lp_n= self.calc_lp(RP+P, RN, whole_test_U, "test") 
        # results,labels, lp_i, lp_n= self.cal_lp_NILP(P,RP, RN, whole_test_U, "test") 
        evaluation(results, labels)
        return lp_i, lp_n
               
    def calc_pu(self, P, U):
    
        RP = []
        Pcopy = P[:]
        Ucopy = U[:]
        print('Calculate ranks...', flush=True)
        top = int((self.lmbda / self.m) * len(P))
        for self.k in range(0, self.m):
            rank = np.zeros(top)
            rank_names = ['' for temp in range(top)]
            min_value = 0
            min_value_id = 0
            for vi in Ucopy:
                Svi = 0
                for vj in Pcopy:
                    Svi += self.W.loc[vi][vj-1]
                Svi /= len(Pcopy)
                if (Svi > min_value):
                    rank_names[min_value_id] = vi
                    rank[min_value_id] = Svi
                    min_value_id = np.argmin(rank)
                    min_value = rank[min_value_id]
            for i in rank_names:
                Pcopy.append(i)
                Ucopy.remove(i)
                RP.append(i)
                U.remove(i)

        bottom = len(RP + P)
        
        rank = [10000 for temp in range(bottom)]
        RN = ['' for temp in range(bottom)]
        max_value = 10000
        max_value_id = 0
        for vi in U:
            Svi = 0
            for vj in (P + RP):
                Svi += self.W.loc[vi][vj-1]
            Svi /= len(P + RP)
            if (Svi < max_value):
                RN[max_value_id] = vi
                rank[max_value_id] = Svi
                max_value_id = np.argmax(rank)
                max_value = rank[max_value_id]
        
        print('Save P, RP and RN')  
        return P, RP, RN
    
    def calc_lp(self, CP, CN, U, class_info_mode = "train"):
    
        class_info_file = self.temp_path + "class_info_" + class_info_mode + ".txt"
        f = open(class_info_file, 'w')  # class_info_file = "./results/class_info_train(test).txt"
        for i in U:
            if i in CP:
                f.write(str(i) + ':class\t1,0\n')

            if i in CN:
                f.write(str(i) + ':class\t0,1\n')
                
            if i not in CP and i not in CN:
                temp_label = {}
                label = [0,0]
               
                i_adj = self.G[str(i)]      
                for key in i_adj:
                    index = int(float(key))        
                    w = i_adj[key]['weight']
                    temp = [0,0]
                    if index in CP:
                        # 1,0
                        temp = [w,0]
                    if index in CN:
                        temp = [0,w]
                    if index in temp_label:
                        temp = temp_label[key]
                    for n in self.G.neighbors(key):
                       
                        temo = [0,0]
                        if int(n) in CP:
                            # 1,0
                            temo = [w,0]
                        if int(n) in CN:
                            temo = [0,w]
                        if int(n) in temp_label:
                            temo = temp_label[key]
                        for m in range(2):
                            temp[m] += temo[m]
                    for k in range(2):
                        label[k] += temp[k]
            
                temp_label['i'] = label
                s = ",".join(str(i) for i in label)
                f.write(str(i) + ':class\t'+s+'\n')     
        print("Write the class info into the txt.")
        f.close()
        
        
        self.test_for_lp = []  
        with open(class_info_file) as f:
            for row in f:
                name, class_vector = row.split('\t')
                id, layer = name.split(':')
                if(layer == 'class'):
                    self.test_for_lp.append(row)

        self.cmn = self.getCMN()
        tp = fp = tn = fn = 0

        lp_i = []
        lp_n = []
        
        results = []
        labels = []
        for f in self.test_for_lp:
            name, class_vector = f.split('\t')
            name, layer = name.split(':')
            if(layer=='class'):
                # label = int(self.labels[self.test_index.index(int(name))])
                label = 0
                label_ = int(self.labels[int(name)-1])   
                if label_ == 0:
                    label = -1
                else:
                    label = 1
                x, y = class_vector.split(',')
                x, y = float(x), float(y)
                
            arg = self.argMaxClass(f=[x,y])
            
            if(arg == 0 and label == 1):
                results.append(1)
                labels.append(1)
                lp_i.append(int(name))

            elif(arg== 0 and label == -1):
                results.append(1)
                labels.append(-1)
                lp_i.append(int(name))

            elif(arg== 1 and label == -1):
                results.append(-1)
                labels.append(-1)
                lp_n.append(int(name))

            elif(arg==1 and label == 1):
                results.append(-1)
                labels.append(1)
                lp_n.append(int(name))

        return results,labels,lp_i,lp_n
          
    def getCMN(self):
        cmn = [0,0]
        rodou = False

        for f in self.test_for_lp:
            name, class_vector = f.split('\t')
            x, y = class_vector.split(',')
            x, y = float(x), float(y)
    
            rodou = True
            f = [x,y]
            for i in range(len(f)):
                cmn[i] += f[i]
        if rodou:
            return cmn
        else:
            return None
        
    def argMaxClass(self, f):
        argMax = -1
        prob = probClasse(f)
        
        if sum(prob) == 0:
            argMax = 1  
        else:
            maxValue = 2.2250738585072014e-308
            for i in range(0, 2):
                value = prob[i] * (f[i] / self.cmn[i])
                if(value > maxValue):
                    argMax = i
                    maxValue = value
        return argMax 
    
    def probclass(self, f):
        if f[0] > f[1]:
            return 1
        else:
            return -1


def probClasse(f):
	s = 0
	p = []
	for v in f:
		s += v
	if s == 0:
		p = [0,0]
	else:
		p = [0,0]
		for i in range(len(f)):
			p[i] = f[i] / s
	return p 
