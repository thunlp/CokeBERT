# CokeBert
[CokeBERT](https://arxiv.org/abs/2009.13964): Contextual Knowledge Selection and Embedding towards Enhanced Pre-Trained Language Models

- EMNLP-Findings 2020 Accepted.

- AI-Open 2021 Accepted.


## Overview
<figure>
<img src=https://github.com/thunlp/CokeBERT/blob/main/CokeBert-1.0/CokeBert.jpg width="80%">
<figcaption>Figure: The example of capturing knowledge contextfrom a KG and incorporating them for language understanding. Different sizes of circles express different entity importance for understanding the given sentence.
</figcaption>
</figure>



## Reqirements:

- Pytorch>=0.4.1
- Python3
- tqdm
- boto3
- requests
- Python 3.6.9
- pytorch 1.2.0
- gcc 7.5.0
- Apex
(If you want to use fp16, you <strong>MUST</strong> make sure the apex commit is 880ab925bce9f817a93988b021e12db5f67f7787.)
We have already provide this version apex in our source code. Please follow the instructions as below:
```
cd apex
python3 setup.py install --user --cuda_ext --cpp_ext
```


## Prepare Pre-trained and Fine-tuned Data

- We will provide dataset for pre-training. If you want to use the latest data, pleas follow the [ERNIE](https://github.com/thunlp/ERNIE "ERNIE") pipline to pre-process your data.
After pre-process Pre-trained data, move them to the corresopinding dir

>BERT-base/RoBERTA-base
```
mv merge.bin DKPLM/data/pretrain_data_bert
mv mergr.idx DKPLM/data/pretrain_data_bert

mv merge.bin DKPLM/data/pretrain_data_roberta
mv mergr.idx DKPLM/data/pretrain_data_roberta
```

- As most datasets except FewRel don not have entity annotations, we use the annotated dataset from ERNIE. Downlaod them from [data](https://drive.google.com/file/d/1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im/view). Then, please unzip and save them (data) to the corresopinding dir.
```
mv data to DKPLM/data/data
```

- Download base models for pre-training
>Roberta/Bert/ERNIE
```
roberta-base (Download roberta_base from Huggieface to DKPLM/data/bert_base)
bert-base-uncased (Download bert_base from Huggieface to DKPLM/data/roberta_base)
ernie_base (Download ernie_base from https://github.com/thunlp/ERNIE to DKPLM/data/ernie_base)
```

- Knowledge Embedding (including entity and relation to id information)
```
Download kg_embed from and move to DKPLM/data/kg_embed
```


- Generate Knowledge Graph Neighbors
```
python3 DKPLM/code/DKPLM_BERTbase/code/knowledge_bert/preprocess_n.py
```


## Pre-train

### DKPLM_BERTbase/DKPLM_RoBERTabase
```
cd DKPLM/code/DKPLM_BERTbase
bash run_pretrain_2layer.sh

cd DKPLM/code/DKPLM_RoBERTabase
bash run_pretrain_2layer.sh
```
You could also directly download the pre-trained CokeBert from here: [Checkpoints](https://drive.google.com/file/d/1Ce7Nq7vJ83l4lOV9SiiN2Kq831z_phsV/view?usp=sharing)


## Fine-tune
- After pre-training DKPLM model, move pytorch_model.bin to the corresponding dir
DKPLM/data/DKPLM_BERTbase_2layer DKPLM/data/DKPLM_RoBERTabase_2layer

### DKPLM_BERTbase
```
cd DKPLM/code/DKPLM_RoBERTabase
```

#### FewRel/Figer/Open Entity/TACRED
```
bash run_fewrel_2layer.sh
bash run_figer_2layer.sh
bash run_open_2layer.sh
bash run_tacred_2layer.sh
```

### DKPLM_RoBERTabase
```
cd DKPLM/code/DKPLM_RoBERTabase
```

#### FewRel/Figer/Open Entity/TACRED
```
bash run_fewrel_2layer.sh
bash run_figer_2layer.sh
bash run_open_2layer.sh
bash run_tacred_2layer.sh
```





<!-- 
### Empirical Analysis
#### DKPLM_BERTbase
```
cd DKPLM/code/DKPLM_RoBERTabase
```
##### FewRel
	###### ERNIE
	bash analysis_fewrel_ernie.sh

	###### DKPLM
	bash analysis_fewrel_DK.sh


##### TACRED
	###### ERNIE
	bash analysis_tacred_ernie.sh

	###### DKPLM
	bash analysis_tacred_DK.sh


#### DKPLM_RoBERTabase
```
cd DKPLM/code/DKPLM_RoBERTabase
```
##### FewRel
	###### ERNIE
	bash analysis_fewrel_ernie.sh

	###### DKPLM
	bash analysis_fewrel_DK.sh

##### TACRED
	###### ERNIE
	bash analysis_tacred_ernie.sh

	###### DKPLM
	bash analysis_tacred_DK.sh
-->

## Citation

Please cite our paper if you use CokeBert in your work:
```
@article{SU2021,
title = {CokeBERT: Contextual Knowledge Selection and Embedding towards Enhanced Pre-Trained Language Models},
author = {Yusheng Su and Xu Han and Zhengyan Zhang and Yankai Lin and Peng Li and Zhiyuan Liu and Jie Zhou and Maosong Sun},
journal = {AI Open},
year = {2021},
issn = {2666-6510},
doi = {https://doi.org/10.1016/j.aiopen.2021.06.004},
url = {https://arxiv.org/abs/2009.13964},
}
```

## Contact
[Yusheng Su](https://yushengsu-thu.github.io/)

Mail: yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn
