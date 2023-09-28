import pandas as pd
import numpy as np
import math
import torch
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataset import MyDataSet
from model import Pathway_Guided_Transformer
from utils import train_one_epoch, train_evaluate
from sklearn.metrics import f1_score, recall_score, precision_score
from collections import Counter
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_integer_mapping(le):
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})
    return res

if __name__ == "__main__":

    train_data_path = 'RNAseq/KEGG/data_train.pkl'
    KEGG_result = 'RNAseq/KEGG_Pathway_information.csv'
    model_save_path = 'Model_weight/'

    data = pd.read_pickle(train_data_path)
    data = data.replace(np.nan, 0)
    print(Counter(data['label']))
 
    x_train = data.iloc[:, :-1].values
    x_train = np.log2(x_train+1)
    y_train = data.iloc[:,-1]
    lbl = LabelEncoder()
    y_train = lbl.fit_transform(y_train)
    print(get_integer_mapping(lbl))

    pathway_df = pd.read_csv(KEGG_result, header=0)
    pathway_num = list(pathway_df['count'])

    batch_size = 64
    epochs = 30
    lr, lrf = 0.0001, 0.001
    max_acc, max_f1, max_recall, max_precision = 0, 0, 0, 0
    test_acc, test_f1, test_recall, test_precision  = [], [], [], []

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    KK = 0
    
    for train, val in kfold.split(x_train, y_train):
        print('*'*30, KK, '*'*30)
        loss_m = np.inf
        train_data_mRNA, train_data_y_label = x_train[train,:], y_train[train]
        val_data_mRNA, val_data_y_label = x_train[val,:], y_train[val]

        train_data_mRNA = train_data_mRNA.reshape((train_data_mRNA.shape[0], 1, train_data_mRNA.shape[1]))
        val_data_mRNA = val_data_mRNA.reshape((val_data_mRNA.shape[0], 1, val_data_mRNA.shape[1]))

        train_data_set_KFold = MyDataSet(train_data_mRNA, train_data_y_label)
        val_data_set_KFold = MyDataSet(val_data_mRNA, val_data_y_label)

        train_loader_KFold = torch.utils.data.DataLoader(train_data_set_KFold, batch_size=batch_size, shuffle=True)
        val_loader_KFold = torch.utils.data.DataLoader(val_data_set_KFold, batch_size=batch_size, shuffle=True)

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

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        criterion = torch.nn.CrossEntropyLoss()
        train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

        for epoch in range(epochs):
            print("-"*50)
            print("epoch:", epoch, "K:", KK)
            train_loss, train_acc = train_one_epoch(now_epoch=epoch, all_epoch=epochs, model=model, optimizer=optimizer, data_loader=train_loader_KFold, 
                                                    num=len(train_data_set_KFold), criterion=criterion, device=device)
            scheduler.step()
            val_loss, val_acc, true_label, pre_label = train_evaluate(model=model, data_loader=val_loader_KFold, 
                                                                      num=len(val_data_set_KFold), criterion=criterion, device=device)

            if val_loss<loss_m:
                loss_m = val_loss
                max_acc = val_acc
                max_f1 = f1_score(true_label, pre_label, average='weighted')
                max_recall = recall_score(true_label, pre_label, average='weighted')
                max_precision = precision_score(true_label, pre_label, average='weighted', zero_division=0)
                torch.save(model.state_dict(), model_save_path + "weights_" + str(KK) + ".pth")

            train_acc_list.append(train_acc.item())
            val_acc_list.append(val_acc.item())
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            print("epoch: {}, train loss: {:.8f}, val loss: {:.8f}, train acc: {:.4f}, val acc: {:.4f}".format(epoch, train_loss, val_loss, train_acc, val_acc))
        
        test_acc.append(max_acc.item())
        test_f1.append(max_f1)
        test_recall.append(max_recall)
        test_precision.append(max_precision)
        print("train_acc_list: ", train_acc_list)
        print('val_acc_list: ', val_acc_list)
        print('train_loss_list: ', train_loss_list)
        print('val_loss_list: ', val_loss_list)
        KK = KK+1

    print('*'*60)
    print("K-fold test_acc: ", test_acc)
    print("K-fold test_f1: ", test_f1)
    print("K-fold test_recall: ", test_recall)
    print("K-fold test_precision: ", test_precision)