import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
import torch
from model import Pathway_Guided_Transformer
from dataset import MyDataSet
from utils import evaluate, evaluate_zhongshan
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

pathway_df = pd.read_csv('/mnt/mydisk/xjj/RNAseq/data/GO_gene/KEGG_result_all611.csv', header=0)
pathway_num = list(pathway_df['count'])

model = Pathway_Guided_Transformer(
        num_classes = 32,
        pathway_number = pathway_num,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 1024,
        dropout = 0.1,
        emb_dropout = 0.1,
    ).to(device)

# TCGA
TCGA_labelDict = {'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'COADREAD': 5, 'DLBC': 6, 'ESCA': 7, 
             'GBM': 8, 'HNSC': 9, 'KICH': 10, 'KIRC': 11, 'KIRP': 12, 'LAML': 13, 'LGG': 14, 'LIHC': 15, 
             'LUAD': 16, 'LUSC': 17, 'MESO': 18, 'OV': 19, 'PAAD': 20, 'PCPG': 21, 'PRAD': 22, 'SARC': 23, 
             'SKCM': 24, 'STAD': 25, 'TGCT': 26, 'THCA': 27, 'THYM': 28, 'UCEC': 29, 'UCS': 30, 'UVM': 31}

# ICGC
ICGC_labelDict = {'PRAD-CA': 22, 'LIRI-JP': 15, 'PACA-AU':20, 'PBCA-US':8, 'BRCA-KR':2, 'PACA-CA':20}

# GEO_metastatic & GEO_metastatic_primary
GEO_metastatic_labelDict = {'cervical':3, 'Gastric':25, 'kidney':11, 'Kidney':11, 'Liver':15, 'Breast':2, 'CRC':5, 
             'skin':24, 'esophagus':7, 'pancrease':20, 'Melanoma':31, 'Thyroid':27, 'ovarian':19}
GEO_metastatic_primary_labelDict = GEO_metastatic_labelDict

# GEO_GSE2109
GEO_GSE2109_labelDict = {'neck':9, 'stomach':25, 'breast':2, 'Renal':11, 'Brain':8, 'lung':16, 'Cervix':3, 'Uterus':29, 'colon':5, 
             'pancreas':20, 'Prostate':22, 'skin':24, 'ovary':19, 'thyroid':27, 'liver':15, 'kidney':11, 'rectum':5, 'bladder':1}

# GPL570
GEO_GPL570_labelDict = {'lung': 16, 'ovarian':19, 'ESCC':7, 'CRC':5, 'CESC':3}

GEO_2109_570_labelDict = {'breast':2, 'Cervix':3, 'Prostate':22, 'ovary':19, 'thyroid':27, 'kidney':11, 'rectum':5, 'lung': 16, 'CRC':5}

# 编号：TCGA标签
TolabelDict = {0: 'ACC', 1: 'BLCA', 2: 'BRCA', 3: 'CESC', 4: 'CHOL', 5: 'COADREAD', 6: 'DLBC', 7: 'ESCA', 8: 'GBM', 
                9: 'HNSC', 10: 'KICH', 11: 'KIRC', 12: 'KIRP', 13: 'LAML', 14: 'LGG', 15: 'LIHC', 16: 'LUAD', 
                17: 'LUSC', 18: 'MESO', 19: 'OV', 20: 'PAAD', 21: 'PCPG', 22: 'PRAD', 23: 'SARC', 24: 'SKCM', 
                25: 'STAD', 26: 'TGCT', 27: 'THCA', 28: 'THYM', 29: 'UCEC', 30: 'UCS', 31: 'UVM'}

# TCGA转移样本验证
def test_TCGA(path_test_data, path_weight, prediction_path):
    # 读取TCGA数据
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0) # 填充缺失值

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [TCGA_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test 
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    print(classification_report(Y_pre_label, Y_true_label, digits=4, zero_division=0))
    print('precision_score', precision_score(Y_pre_label, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label, Y_true_label, average='weighted'))


    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    cal_TCGA_acc(prediction_path)
    
def cal_TCGA_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'SARC':0, 'PCPG':0, 'ESCA':0, 'SKCM':0, 'COADREAD':0, 'PAAD':0, 'PRAD':0, 'HNSC':0, 'CESC':0, 'THCA':0, 'BRCA':0}
    cancer_num_correct ={'SARC':0, 'PCPG':0, 'ESCA':0, 'SKCM':0, 'COADREAD':0, 'PAAD':0, 'PRAD':0, 'HNSC':0, 'CESC':0, 'THCA':0, 'BRCA':0}
    
    multi_label = {'SARC': ['SARC'], 'PCPG': ['PCPG'], 'ESCA': ['ESCA'], 'SKCM': ['SKCM'], 'COADREAD': ['COADREAD'], 'PAAD': ['PAAD'], 
                   'PRAD': ['PRAD'], 'HNSC': ['HNSC'], 'CESC': ['CESC'], 'UCEC': ['UCEC'], 'THCA': ['THCA'], 
                    'BRCA': ['BRCA'], }
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("每个癌症的样本数量")
    print(Counter(predict_data['Y_true_label']))
    print("样本总数量")
    print(predict_num)
    print("预测正确的样本总数量")
    print(correct_num)
    print("在每个癌型中预测正确的样本数量")
    print(cancer_num_correct)
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("每个癌型预测的正确率")
    print(per_cancer_acc)
    print("平均正确率")
    print("acc: ", correct_num/predict_num)

def test_ICGC(path_test_data, path_weight):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0) # 填充缺失值
    print(test_data.shape)
    print(Counter(test_data['label']))

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [ICGC_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
 
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)
    print(classification_report(y_test, pre_label, digits=4, zero_division=0))
    print('precision_score', precision_score(y_test, pre_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(y_test, pre_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(y_test, pre_label, average='weighted'))
    print('test_acc', test_acc.item())

def test_zhongshan(path_test_data, path_weight, prediction_path):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0) # 填充缺失值

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [TCGA_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
    
    model.load_state_dict(torch.load(path_weight))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label, pre_label_top5, pre_label_top3 = evaluate_zhongshan(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)
    print('pre_label_top5')
    print(pre_label_top5[0])
    label_top1 = []
    label_top2 = []
    label_top3 = []
    label_top4 = []
    label_top5 = []
    for label_5 in pre_label_top5[0]:
        label_top1.append(TolabelDict[label_5[0]])
        label_top2.append(TolabelDict[label_5[1]])
        label_top3.append(TolabelDict[label_5[2]])
        label_top4.append(TolabelDict[label_5[3]])
        label_top5.append(TolabelDict[label_5[4]])

    print(classification_report(y_test, pre_label))
    print('test_acc', test_acc.item())

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    df_2 = pd.DataFrame({'Y_true_label':Y_true_label, 'Y_pre_label_1':label_top1, 'Y_pre_label_2':label_top2, 'Y_pre_label_3':label_top3, 'Y_pre_label_4':label_top4, 'Y_pre_label_5':label_top5})
    df_2.to_csv('/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/prediction/zhongshan_prediction_top_k.csv')

    top_5_correct_num = 0
    top_3_correct_num = 0
    for i in range(len(Y_true)):
        if Y_true[i] in pre_label_top5[0][i]:
            top_5_correct_num = top_5_correct_num + 1
        if Y_true[i] in pre_label_top3[0][i]:
            top_3_correct_num = top_3_correct_num + 1
    print('top5_acc:', top_5_correct_num/len(Y_true))
    print('top3_acc:', top_3_correct_num/len(Y_true))


def test_GEO_metastatic(path_test_data, path_weight, prediction_path):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data[test_data['label'] != 'esophagus']
    test_data = test_data.replace(np.nan, 0) # 填充缺失值

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_metastatic_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)

    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]


    # 计算Precision, Recall, F1-score
    Y_pre_label_change = []
    find_label_list = ['COADREAD', 'BRCA', 'LIHC', 'PAAD', 'STAD', 'ESCA']
    find_label_dict = {'COADREAD':'CRC', 'BRCA':'Breast', 'LIHC':'Liver', 'PAAD': 'pancrease', 'STAD': 'Gastric', 'ESCA': 'Gastric'}
    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)
    
    print(classification_report(Y_pre_label_change, Y_true_label, digits=4, zero_division=0))
    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label,average='weighted'))

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    cal_GEO1_acc(prediction_path)

def cal_GEO1_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'CRC': 0, 'Breast': 0, 'Liver': 0, 'pancrease': 0, 'Gastric': 0}
    cancer_num_correct = {'CRC': 0, 'Breast': 0, 'Liver': 0, 'pancrease': 0, 'Gastric': 0}

    multi_label = {'CRC': ['COADREAD'],'Breast': ['BRCA'],'Liver': ['LIHC'],'pancrease': ['PAAD'], 'Gastric': ['STAD', 'ESCA']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("每个癌症的样本数量")
    print(Counter(predict_data['Y_true_label']))
    print("样本总数量")
    print(predict_num)
    print("预测正确的样本总数量")
    print(correct_num)
    print("在每个癌型中预测正确的样本数量")
    print(cancer_num_correct)
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("每个癌型预测的正确率")
    print(per_cancer_acc)
    print("平均正确率")
    print("acc: ", correct_num/predict_num)

def test_GEO_metastatic_primary(path_test_data, path_weight, prediction_path):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0) # 填充缺失值
    print(test_data.shape)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_metastatic_primary_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
  
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    # 计算Precision, Recall, F1-score
    Y_pre_label_change = []
    find_label_list = ['COADREAD', 'SKCM', 'UVM', 'KIRC', 'KIRP', 'KICH', 'CESC', 'OV']
    find_label_dict = {'COADREAD':'CRC', 'SKCM':'Melanoma', 'UVM':'Melanoma', 'KIRC':'kidney', 'KIRP':'kidney', 'KICH':'kidney', 'CESC':'cervical', 'OV':'ovarian'}

    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)
    
    print(classification_report(Y_pre_label_change, Y_true_label, digits=4, zero_division=0))
    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label, average='weighted'))

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    cal_GEO2_acc(prediction_path)

def cal_GEO2_acc(predicition):
    predict_data = pd.read_csv(predicition)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'CRC': 0, 'Melanoma': 0, 'kidney': 0, 'cervical': 0, 'ovarian': 0}
    cancer_num_correct = {'CRC': 0, 'Melanoma': 0, 'kidney': 0, 'cervical': 0, 'ovarian': 0}

    multi_label = {'CRC': ['COADREAD'],'Melanoma': ['SKCM', 'UVM'],'kidney': ['KIRC', 'KIRP', 'KICH'],'cervical': ['CESC'], 'ovarian': ['OV']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("每个癌症的样本数量")
    print(Counter(predict_data['Y_true_label']))
    print("样本总数量")
    print(predict_num)
    print("预测正确的样本总数量")
    print(correct_num)
    print("在每个癌型中预测正确的样本数量")
    print(cancer_num_correct)
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c],4)
    print("每个癌型预测的正确率")
    print(per_cancer_acc)
    print("平均正确率")
    print("acc: ", correct_num/predict_num)

def test_2109_570(path_test_data, path_weight, prediction_path):
    test_data = pd.read_pickle(path_test_data)
    test_data = test_data.replace(np.nan, 0) # 填充缺失值
    print(test_data.shape)

    x_test = test_data.iloc[:, :-1].values
    x_test = np.log2(x_test+1)
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    y_test_0 = test_data.iloc[:,-1]
    y_test = [GEO_2109_570_labelDict[i] for i in y_test_0]

    test_data_set = MyDataSet(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size)
    
    model.load_state_dict(torch.load(path_weight))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, pre_label = evaluate(model=model, data_loader=test_loader, num=len(test_data_set), criterion=criterion, device=device)

    Y_true = y_test
    Y_true_label = test_data['label']

    Y_pre = pre_label
    Y_pre_label = [TolabelDict[i] for i in Y_pre]

    print(set(Y_pre_label))

    # 计算Precision, Recall, F1-score
    Y_pre_label_change = []
    find_label_list = ['BRCA', 'CESC', 'PRAD', 'OV', 'THCA', 'KIRC', 'KIRP', 'KICH', 'LUAD', 'LUSC', 'COADREAD']
    find_label_dict = {'BRCA':'breast', 'CESC':'Cervix', 'PRAD':'Prostate', 'OV':'ovary', 'THCA': 'thyroid', 'KIRC':'kidney', 
                       'KIRP':'kidney', 'KICH':'kidney', 'LUAD':'lung', 'LUSC':'lung', 'COADREAD':'CRC'}

    for i in Y_pre_label:
        if i in find_label_list:
            i = find_label_dict[i]
        Y_pre_label_change.append(i)
    
    print(classification_report(Y_pre_label_change, Y_true_label, digits=4, zero_division=0))
    print('precision_score', precision_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('recall_score', recall_score(Y_pre_label_change, Y_true_label, average='weighted', zero_division=0))
    print('f1_score', f1_score(Y_pre_label_change, Y_true_label,  average='weighted'))

    df = pd.DataFrame({'Y_true':Y_true, 'Y_true_label':Y_true_label, 'Y_pre':Y_pre, 'Y_pre_label':Y_pre_label})
    df.to_csv(prediction_path)
    cal_2109_570_acc(prediction_path)

def cal_2109_570_acc(prediction_path):
    predict_data = pd.read_csv(prediction_path)
    predict_num = predict_data.shape[0]
    correct_num = 0
    cancer_num = {'breast': 0, 'Cervix': 0, 'Prostate': 0, 'ovary': 0, 'thyroid': 0, 
    'kidney': 0,  'lung': 0, 'CRC':0}
    cancer_num_correct = {'breast': 0, 'Cervix': 0, 'Prostate': 0, 'ovary': 0, 'thyroid': 0, 
    'kidney': 0, 'lung': 0, 'CRC':0}

    multi_label = {'breast': ['BRCA'], 'Cervix': ['CESC'], 'Prostate': ['PRAD'], 'ovary': ['OV'], 'thyroid': ['THCA'], 
    'kidney': ['KIRC', 'KIRP', 'KICH'], 'lung': ['LUAD', 'LUSC'], 'CRC':['COADREAD']}
    
    for i in range(predict_num):
        for cancer_kind in cancer_num.keys():
            if predict_data.at[i, 'Y_true_label']==cancer_kind:
                T_label = multi_label[cancer_kind]
                cancer_num[predict_data.at[i, 'Y_true_label']] = cancer_num[predict_data.at[i, 'Y_true_label']] + 1
                if predict_data.at[i, 'Y_pre_label'] in T_label:
                    correct_num = correct_num+1
                    cancer_num_correct[predict_data.at[i, 'Y_true_label']] = cancer_num_correct[predict_data.at[i, 'Y_true_label']]+1

    print("每个癌症的样本数量")
    print(Counter(predict_data['Y_true_label']))
    print("样本总数量")
    print(predict_num)
    print("预测正确的样本总数量")
    print(correct_num)
    print("在每个癌型中预测正确的样本数量")
    print(cancer_num_correct)
    cancer_list = list(cancer_num_correct.keys())
    per_cancer_acc = {}
    for c in cancer_list:
        per_cancer_acc[c]=round(cancer_num_correct[c]/cancer_num[c], 4)
    print("每个癌型预测的正确率")
    print(per_cancer_acc)
    print("平均正确率")
    print("acc: ", correct_num/predict_num)

# 测试数据路径
test_data_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/KEGG_RNA611/'
path_test_data_TCGA = test_data_path + 'RNAseq_TCGA_metastatic.pkl'
path_test_data_ICGC = test_data_path + 'RNAseq_ICGC_metastatic.pkl'
path_test_data_zhongshan = test_data_path + 'RNAseq_zhongshan_metastatic_37.pkl'
path_test_data_GEO1 = test_data_path + 'RNAseq_GEO_metastatic.pkl'
path_test_data_GEO2 = test_data_path + 'RNAseq_GEO_metastatic_primary.pkl'
path_test_data_2109_570 = test_data_path + 'RNAseq_2109_570.pkl'

# 模型权重文件
path_weight = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/model_weights/new_KEGG_transformer_weights9.pth'

# 预测标签保存
prediction_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/prediction/'
TCGA_prediction = prediction_path + 'TCGA_prediction.csv'
zhongshan_prediction = prediction_path + 'zhongshan_prediction.csv'
GEO_metastatic_prediction = prediction_path + 'GEO1_prediction.csv'
GEO_metastatic_primary_prediction = prediction_path + 'GEO2_prediction.csv'
GEO_2109_570_prediction = prediction_path + '2109_570_prediction.csv'


# print("="*20)
# print("TCGA验证：")
# test_TCGA(path_test_data_TCGA, path_weight, TCGA_prediction)
# print("="*20)
# print("ICGC验证：")
# test_ICGC(path_test_data_ICGC, path_weight)
print("="*20)
print("zhongshan验证：")
test_zhongshan(path_test_data_zhongshan, path_weight, zhongshan_prediction)
# print("="*20)
# print('GEO_metastatic验证：')
# test_GEO_metastatic(path_test_data_GEO1, path_weight, GEO_metastatic_prediction)
# print("="*20)
# print('GEO_metastatic_primary验证：')
# test_GEO_metastatic_primary(path_test_data_GEO2, path_weight, GEO_metastatic_primary_prediction)
# print("="*20)
# print('GPL2109-570验证：')
# test_2109_570(path_test_data_2109_570, path_weight, GEO_2109_570_prediction)

# [3] 1013367
 
