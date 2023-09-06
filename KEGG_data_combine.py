import pandas as pd
data = pd.read_csv("/mnt/mydisk/xjj/RNAseq/data/GO_gene/KEGG_result_all611.csv", header=0)

print(data['count'].sum())
all_gene_set = []
all_gene_list = []
for gene_set in data['gene_name']:
    all_gene_set.append(gene_set.split("/"))
    all_gene_list.extend(gene_set.split("/"))
print(len(all_gene_list))
print(len(set(all_gene_list)))

base_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/check_data/KEGG_RNA/'
# 新保存路径
new_path = '/mnt/mydisk/chenying/CUP-xjj/KEGG_Transformer/KEGG_RNA611/'
filelist = ['data_train.pkl', 'RNAseq_TCGA_metastatic.pkl', 'RNAseq_ICGC_metastatic.pkl', 'RNAseq_zhongshan_metastatic_37.pkl',
            'RNAseq_GEO_metastatic.pkl', 'RNAseq_GEO_metastatic_primary.pkl', 'RNAseq_2109_570.pkl', 'RNAseq_TCGA_primary.pkl']

for filename in filelist:
    print(filename)
    data_RNA = pd.read_pickle(base_path + filename)

    KEGG_data_dict = dict()
    gene_num = 1
    for gene in all_gene_list:
        KEGG_data_dict['gene'+str(gene_num)] = data_RNA[gene]
        gene_num = gene_num + 1
        
    KEGG_data_dict['label'] = data_RNA['label']
    data_KEGG = pd.DataFrame(KEGG_data_dict)
    print(data_KEGG)
    data_KEGG.to_pickle(new_path + filename)