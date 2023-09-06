import pandas as pd
import numpy as np
import math
import sys
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataset import MyDataSet
from model import Pathway_Guided_Transformer
from utils import train_one_epoch, train_evaluate
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import Counter
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取标签映射关系
def get_integer_mapping(le):
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})
    return res

if __name__ == "__main__":

    train_data_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/KEGG_RNA611/data_train.pkl'
    KEGG_result = '/mnt/mydisk/xjj/RNAseq/data/GO_gene/KEGG_result_all611.csv'
    model_save_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/'
    # 读取训练数据
    data = pd.read_pickle(train_data_path)
    data = data.replace(np.nan, 0)
    print(Counter(data['label']))
 
    x_train = data.iloc[:, :-1].values
    x_train = np.log2(x_train+1) # log2转换
    y_train = data.iloc[:,-1]
    lbl = LabelEncoder()
    y_train = lbl.fit_transform(y_train)
    print(get_integer_mapping(lbl))

    # 通道信息
    pathway_df = pd.read_csv(KEGG_result, header=0)
    pathway_num = list(pathway_df['count'])

    batch_size = 64
    epochs = 15
    lr = 0.0001
    lrf = 0.001

    # 10折交叉验证
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    test_acc = []
    test_f1 = []
    test_recall = []
    test_precision = []
    test_roc = []

    max_acc = 0
    max_f1 = 0
    max_recall = 0
    max_precision = 0
    max_roc = 0
    KK = 0
    

    for train, val in kfold.split(x_train, y_train):
        loss_m = np.inf

        train_data_mRNA, train_data_y_label = x_train[train,:], y_train[train]
        val_data_mRNA, val_data_y_label = x_train[val,:], y_train[val]

        train_data_mRNA = train_data_mRNA.reshape((train_data_mRNA.shape[0], 1, train_data_mRNA.shape[1]))
        val_data_mRNA = val_data_mRNA.reshape((val_data_mRNA.shape[0], 1, val_data_mRNA.shape[1]))

        train_data_set_KFold = MyDataSet(train_data_mRNA, train_data_y_label)
        val_data_set_KFold = MyDataSet(val_data_mRNA, val_data_y_label)

        train_loader_KFold = torch.utils.data.DataLoader(train_data_set_KFold, batch_size=batch_size, shuffle=True)
        val_loader_KFold = torch.utils.data.DataLoader(val_data_set_KFold, batch_size=batch_size, shuffle=True)
        print("train_data_set, len:{}".format(len(train_data_set_KFold)))
        print("val_data_set, len:{}".format(len(val_data_set_KFold)))

        model = Pathway_Guided_Transformer(
                num_classes = 32,
                pathway_number = pathway_num,
                dim = 10,
                depth = 6,
                heads = 8,
                mlp_dim = 1024,
                dropout = 0.1,
                emb_dropout = 0.1,
            ).to(device)
        
        # print(model)

        # summary(model, input_size=(1, train_data_mRNA.shape[2]))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        criterion = torch.nn.CrossEntropyLoss()
        train_acc_list = []
        val_acc_list = []
        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):

            print("-"*50)
            print("epoch:", epoch, "K:", KK)
            train_loss, train_acc = train_one_epoch(now_epoch=epoch, all_epoch=epochs, model=model, optimizer=optimizer, data_loader=train_loader_KFold, num=len(train_data_set_KFold), criterion=criterion, device=device)

            scheduler.step()
            val_loss, val_acc, true_label, pre_label = train_evaluate(model=model, data_loader=val_loader_KFold, num=len(val_data_set_KFold), criterion=criterion, device=device)

            if val_loss<loss_m:
                loss_m = val_loss
                max_acc = val_acc
                max_f1 = f1_score(true_label, pre_label, average='weighted')
                max_recall = recall_score(true_label, pre_label, average='weighted')
                max_precision = precision_score(true_label, pre_label, average='weighted')
                torch.save(model.state_dict(), model_save_path + "new_KEGG_transformer_weights" + str(KK) + ".pth")

            train_acc_list.append(train_acc.item())
            val_acc_list.append(val_acc.item())
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print("epoch: {}, train loss: {:.8f}, val loss: {:.8f}, train acc: {:.4f}, val acc: {:.4f}".format(epoch+1, train_loss, val_loss, train_acc, val_acc))
        
        test_acc.append(max_acc.item())
        test_f1.append(max_f1)
        test_recall.append(max_recall)
        test_precision.append(max_precision)
        KK = KK+1
        print("train_acc_list")
        print(train_acc_list)
        print('val_acc_list')
        print(val_acc_list)
        print('train_loss_list')
        print(train_loss_list)
        print('val_loss_list')
        print(val_loss_list)
        
    print(test_acc)
    print("平均test_acc", np.mean(test_acc))
    print(test_f1)
    print("平均test_f1", np.mean(test_f1))
    print(test_recall)
    print("平均recall", np.mean(test_recall))
    print(test_precision)
    print("平均test_precision", np.mean(test_precision))