from tadpak import pak
from tadpak import evaluate
from metrics.metrics import * 


import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def detection_adjustment(results,labels):

    pred = []
    gt = []
    for item in results:
        pred.append(item)
    for item in labels:
        gt.append(item)

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1): 
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred, gt


def evaluation(results, labels):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    import pandas as pd
    matriz_confusao = pd.DataFrame(0, columns=['pred_pos', 'pred_neg'],index=['classe_pos', 'classe_neg'])
    for l in range(len(results)):
        rotulo_original =  'classe_pos' if labels[l] == 1 else 'classe_neg'
        predito = 'pred_pos' if results[l] == 1 else 'pred_neg'
        matriz_confusao[predito][rotulo_original] += 1
				
    print(matriz_confusao)
		

    # matriz_confusao.to_csv('out/confusion_lphn_{}.csv'.format(1), sep='\t')
    positive_total = matriz_confusao['pred_pos'].sum() #total de noticias classificadas como positivo
    true_positive_news = matriz_confusao['pred_pos'].loc['classe_pos'].sum()
    print("======================Evaluation===========================")
    f = open('avaliados.csv', 'a')
    TP = matriz_confusao['pred_pos'].loc['classe_pos'].sum()
    FP = matriz_confusao['pred_pos'].loc['classe_neg'].sum()    
    TN = matriz_confusao['pred_neg'].loc['classe_neg'].sum()
    FN = matriz_confusao['pred_neg'].loc['classe_pos'].sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2*precision*recall)/(precision+recall)
    f.write("-------------------------\n")
    f.write("\t".join([str(f1_score(labels, results, average='macro')), str(f1_score(labels, results, average='micro')), 
			str(accuracy_score(labels, results)), str(precision_score(labels, results, average='macro')), 
			str(precision_score(labels, results, average='micro')), str(recall_score(labels, results, average='macro')), 
			str(recall_score(labels, results, average='micro')), str(true_positive_news/positive_total), 
			str(precision), str(recall), str(f1)])+'\n')

    f.close()
		
    print('Macro f1:', f1_score(labels, results, average='macro'))
    print('Micro f1:', f1_score(labels, results, average='micro'))
    print('accuracy:', accuracy_score(labels, results))
    print('Macro Precision:', precision_score(labels, results, average='macro'))
    print('Micro Precision:', precision_score(labels, results, average='micro'))
    print('Macro Recall:', recall_score(labels, results, average='macro'))
    print('Micro Recall:', recall_score(labels, results, average='micro'))
    print('True Positive:', true_positive_news/positive_total)
    print('Interest-class precision: ', precision)
    print('Interest-class recall:', recall)
    print('Interest-class f1:', f1)
    return f1, precision, recall
 

def pak_auc(results,labels):
    scores = np.array(results) 
    targets = np.array(labels)
    thres = 0.5

    candidate = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    f1 = []
    for k in candidate:
        adjusted_preds = pak.pak(scores, targets, thres, k)  
        # print(adjusted_preds)
        # results = evaluate.evaluate(scores, targets)456
        
        # results = evaluate.evaluate(adjusted_preds, targets)
        for item in range(len(adjusted_preds)):
            if adjusted_preds[item] == False:
                adjusted_preds[item] = 0
            else:
                adjusted_preds[item] = 1
        # 计算f1
        precision, recall, threshold = metrics.precision_recall_curve(targets, adjusted_preds)
        f1_score = 2 * precision * recall / (precision + recall + 1e-12)
        f1.append(np.max(f1_score))
    # print("K range(0,1) f1:",f1)
    auc = sum(f1)/len(f1)
    return auc


def f1_prec_recall_PA(preds, gts, k=0):
    # Find out the indices of the anomalies
    gt_idx = np.where(gts == 1)[0]
    anomalies = []
    new_preds = np.array(preds)
    # Find the continuous index 
    for  _, g in groupby(enumerate(gt_idx), lambda x : x[0] - x[1]):
        anomalies.append(list(map(itemgetter(1), g)))
    # For each anomaly (point or seq) in the test dataset
    for a in anomalies:
        # Find the predictions where the anomaly falls
        pred_labels = new_preds[a]
        # Check how many anomalies have been predicted (ratio)
        if len(np.where(pred_labels == 1)[0]) / len(a) > (k/100):
            # Update the whole prediction range as correctly predicted
            new_preds[a] = 1
    f1_pa = f1_score(gts, new_preds)
    prec_pa = precision_score(gts, new_preds)
    recall_pa = recall_score(gts, new_preds)
    return f1_pa, prec_pa, recall_pa


def f1_prec_recall_K(preds, gts):
    f1_pa_k = []
    prec_pa_k = []
    recall_pa_k = []
    for k in range(0,101):
        f1_pa, prec_pa, recall_pa = f1_prec_recall_PA(preds, gts, k)   
        f1_pa_k.append(f1_pa)
        prec_pa_k.append(prec_pa)
        recall_pa_k.append(recall_pa)
    f1_pak_auc = np.trapz(f1_pa_k)
    prec_pak_auc = np.trapz(prec_pa_k)
    recall_pak_auc = np.trapz(recall_pa_k)
    return f1_pak_auc/100, prec_pak_auc/100, recall_pak_auc/100
    

def get_all_res(results, labels):
 
    f_score, precision, recall = evaluation(results,labels)
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))
    

    f1_pak_auc, prec_pak_auc, recall_pak_auc = f1_prec_recall_K(results, labels)
    print("Precision pak : {:0.4f}, Recall pak : {:0.4f}, F-score pak : {:0.4f} ".format(prec_pak_auc, recall_pak_auc, f1_pak_auc))
    
    matrix = [137]
    scores_simple = combine_all_evaluation_scores(np.array(results), np.array(labels))
    for key, value in scores_simple.items():
        matrix.append(value)
        print('{0:21} : {1:0.4f}'.format(key, value))

    
    pred, gt = detection_adjustment(results,labels)
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(gt, pred)
    pa_precision, pa_recall, pa_f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, pa_precision, pa_recall, pa_f_score))
    return f_score, precision, recall, f1_pak_auc, prec_pak_auc, recall_pak_auc, pa_f_score, pa_precision, pa_recall

    
def cal_mean_matrix(results_list):
    res = np.mean(np.array(results_list), axis=0)
    print("===================== Average Results =====================")
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(res[1], res[2], res[0]))
    print("Precision Pak : {:0.4f}, Recall Pak : {:0.4f}, F-score Pak : {:0.4f} ".format(res[4], res[5], res[3]))
    print("Precision Pa : {:0.4f}, Recall Pa : {:0.4f}, F-score Pa : {:0.4f} ".format(res[7], res[8], res[6]))
    print("===========================================================")
    
    
    