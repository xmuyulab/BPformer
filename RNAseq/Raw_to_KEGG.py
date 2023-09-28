import pandas as pd

base_path = 'RNAseq/Raw/'
new_path = 'RNAseq/KEGG/'
filelist = ['data_train.pkl', 'GEO-RNA-m.pkl', 'ICGC-RNA-p-m.pkl', 'ICGC-RNA-p.pkl', 'GEO-RNA-p-m.pkl', 'TCGA-RNA-p.pkl', 'GEO-RNA-p.pkl', 'TCGA-RNA-p-m.pkl']

KEGG_info = pd.read_csv("RNAseq/KEGG_Pathway_information.csv", header=0)

all_gene_list = []
for gene_set in KEGG_info['gene_name']:
    all_gene_list.extend(gene_set.split("/"))

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
