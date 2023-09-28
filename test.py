import pandas as pd
import torch
from model import Pathway_Guided_Transformer
from utils import test_TCGA, test_ICGC, test_GEO_p_m, test_GEO_m, test_GEO_p

# Test data path
test_data_path = 'RNAseq/KEGG/'
path_test_data_TCGA = test_data_path + 'TCGA-RNA-p-m.pkl'
path_test_data_ICGC = test_data_path + 'ICGC-RNA-p-m.pkl'
path_test_data_GEO_m = test_data_path + 'GEO-RNA-m.pkl'
path_test_data_GEO_p_m = test_data_path + 'GEO-RNA-p-m.pkl'
path_test_data_GEO_p = test_data_path + 'GEO-RNA-p.pkl'

# Model weight file
path_weight = 'best_wight.pth'

# prediction result
prediction_path = 'Prediction_result/'
TCGA_prediction = prediction_path + 'TCGA_prediction.csv'
ICGC_prediction = prediction_path + 'ICGC_prediction.csv'
GEO_metastatic_prediction = prediction_path + 'GEO_m_rediction.csv'
GEO_metastatic_primary_prediction = prediction_path + 'GEO_p_m_prediction.csv'
GEO_primary_prediction = prediction_path + 'GEO_p_prediction.csv'

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
batch_size = 64
pathway_df = pd.read_csv('RNAseq/KEGG_Pathway_information.csv', header=0)
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

print("*"*20, "test for TCGA", "*"*20)
TCGA_acc = test_TCGA(path_test_data_TCGA,  path_weight, TCGA_prediction, model, batch_size, device)
print("*"*20, "test for ICGC", "*"*20)
ICGC_acc = test_ICGC(path_test_data_ICGC, path_weight, ICGC_prediction, model, batch_size, device)
print("*"*20, "test for GEO_metastatic", "*"*20)
GEO_m_acc = test_GEO_m(path_test_data_GEO_m, path_weight, GEO_metastatic_prediction, model, batch_size, device)
print("*"*20, "test for GEO_metastatic_primary", "*"*20)
GEO_p_m_acc = test_GEO_p_m(path_test_data_GEO_p_m,  path_weight, GEO_metastatic_primary_prediction, model, batch_size, device)
print("*"*20, "test for GEO_primary", "*"*20)
GEO_p_acc = test_GEO_p(path_test_data_GEO_p, path_weight, GEO_primary_prediction, model, batch_size, device)

print("*"*50)
print("TCGA_acc, ICGC_acc, GEO_m_acc, GEO_p_m_acc, GEO_p_acc")
print([TCGA_acc, ICGC_acc, GEO_m_acc, GEO_p_m_acc, GEO_p_acc])
