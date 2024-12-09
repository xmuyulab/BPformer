# BPformer: Tracing Unknown Tumor Origins with a Biological Pathway-based Transformer Model

  ![image](https://github.com/xmuyulab/BPformer/blob/main/img/BPformer.png)

## Introduction of BPformer

In this study, we developed BPformer, a novel deep learning method for tracing the tumor origins of CUP patients that integrates the Transformer model with prior knowledge of 335 biological pathways from the KEGG (Kyoto Encyclopedia of Genes and Genomes) database. This classifier was trained on transcriptomes of 10,410 primary tumors spanning 32 cancer types and tested on three types of independent test datasets, including primary tumors, primary and metastatic sites of metastatic tumors. For clinical practice, BPformer was also validated within our in-house generated CUP samples. Furthermore, BPformer model was comparatively evaluated against four other cancer origin tracing methods.

## Installation
### Dependencies
```
Python 3.8.13
PyTorch == 2.0.1
scikit_learn
einops
numpy
pandas
tqdm
```

a. Create a conda virtual environment and activate it.

```shell
conda create -n bpformer python=3.8.13
git clone https://github.com/xmuyulab/BPformer.git
cd BPformer
conda activate bpformer
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

c. Next step is to install a set of required python packages, including scikit_learn, einops, numpy, pandas, tqdm.

```shell
pip install -r requirements.txt
```

## Preparation of input data
a. Download the RNAseq data and put them under the path 'RNAseq/Raw'. We provide our experiment dataset in https://drive.google.com/drive/folders/1N8iz37klzR6MJfboG-CrAuLDaonsEHIJ.

b. Transfer the input RNAseq data into required format (genes are arranged in biological pathways).

```shell
python RNAseq/Raw_to_KEGG.py
```

## Model training

Our training datasets data_train.pkl included the TCGA and ICGC primary tumors (TCGA-RNA-p.pkl and ICGC-RNA-p.pkl).

```shell
python train.py
```

## Model inference

BPformer was evaluated in the independent test datasets including the primary tumors from GEO (GEO-RNA-p.pkl), metastatic sites of metastatic tumors from GEO (GEO-RNA-m.pkl), the primary sites of metastatic tumors from GEO, TCGA and ICGC (GEO-RNA-p-m.pkl, TCGA-RNA-p-m.pkl and ICGC-RNA-p-m.pkl).

Our trained model file is saved in best_wight.pth, which can be download in https://drive.google.com/drive/folders/1N8iz37klzR6MJfboG-CrAuLDaonsEHIJ.

```shell
python test.py
```

## License & Usage

If you find our work useful in your research, please consider citing our paper at:

```
@article{xie2024tracing,
  title={Tracing unknown tumor origins with a biological-pathway-based transformer model},
  author={Xie, Jiajing and Chen, Ying and Luo, Shijie and Yang, Wenxian and Lin, Yuxiang and Wang, Liansheng and Ding, Xin and Tong, Mengsha and Yu, Rongshan},
  journal={Cell Reports Methods},
  volume={4},
  number={6},
  year={2024},
  publisher={Elsevier}
}
```



Please open an issue or contact xiejiajing@stu.xmu.edu.cn or cying2023@stu.xmu.edu.cn with any questions.
