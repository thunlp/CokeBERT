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


## Version
#### v1.0
- [x] CokeBert
- [x] CokeRoberta
#### v2.0
- [x] CokeBert
- [ ] CokeRoberta (will release soon)  

## Reqirements:
- pytorch
- transformers
- tqdm
- boto3
- requests
<!--- Apex
(If you want to use fp16, you <strong>MUST</strong> make sure the apex commit is 880ab925bce9f817a93988b021e12db5f67f7787.)
We have already provide this version apex in our source code. Please follow the instructions as below:
```
cd apex
python3 setup.py install --user --cuda_ext --cpp_ext
```
-->

## How to Use
- You need to download Knowledge Embedding (including entity and relation to id information) and knowledge graph neighbor information from [here](https://drive.google.com/drive/folders/116FG9U-U4r674dgfBBL3qMceGXTTcaxc?usp=sharing). Put them in the `data/pretrain` folder and unzip them.
```bash
cd data/pretrain

# Download the files

tar zxvf kg_embed.tar.gz
tar zxvf kg_neighbor.tar.gz

cd ../..
```

- Then, you could obtain pre-trained checkpoints from [here](https://huggingface.co/yushengsu/CokeBERT) and directly use CokeBert.
```python
from coke import CokeBertForPreTraining

model = CokeBertForPreTraining.from_pretrained('checkpoint/coke-bert-base-uncased', neighbor_hop=2)
```

- If you want to pre-train CokeBert with different corpus and knowledge graphs, you could read the following instructions.

## Pre-training

### Prepare Pre-training Data

- Go to the folder for the latest version. Choose a backbone model, e.g. `bert-base-uncased`
```bash
cd CokeBert-2.0-latest
```

- We will provide dataset for pre-training. If you want to use the latest data, pleas follow the [ERNIE](https://github.com/thunlp/ERNIE "ERNIE") pipline to pre-process your data, using the corresponding tokenizer of the backbone model. The outputs are `merbe.bin` and `merge.idx`. After pre-process Pre-trained data, move them to the corresopinding directory.

```bash
export BACKBONE=bert-base-uncased
export HOP=2

mkdir data/pretrain/$BACKBONE

mv merge.bin data/pretrain/$BACKBONE
mv mergr.idx data/pretrain/$BACKBONE
```

- Download the backbone model checkpoint from [Huggingface](https://huggingface.co/models), and move it to the corresponding checkpoint folder for pre-training. Note do not download the `config.json` for the backbone model, since we will be using the config of `coke`.

```bash
wget https://huggingface.co/$BACKBONE/resolve/main/vocab.txt -O checkpoint/coke-$BACKBONE/vocab.txt
wget https://huggingface.co/$BACKBONE/resolve/main/pytorch_model.bin -O checkpoint/coke-$BACKBONE/pytorch_model.bin
```

- Knowledge Embedding (including entity and relation to id information) and knowledge graph neighbor information from [here](https://drive.google.com/drive/folders/116FG9U-U4r674dgfBBL3qMceGXTTcaxc?usp=sharing). Put them in the `data/pretrain` folder and unzip them.

```bash
cd data/pretrain

# Download the files

tar zxvf kg_embed.tar.gz
tar zxvf kg_neighbor.tar.gz

cd ../..
```

- (*Optional*) Generate Knowledge Graph Neighbors. We have provided this data. If you want to change the max number of neighbors, you can run this code to get the new `kg_neighbor` data
```bash
cd data/pretrain
python3 preprocess_n.py
```


### Excute Pre-training

```bash
cd examples
bash run_pretrain.sh
```

It will write log and checkpoint to `./outputs`. Check `src/coke/training_args.py` for more arguments.



## Fine-tuning 

### Fine-tuning Data
- As most datasets except FewRel don not have entity annotations, we use the annotated dataset from ERNIE. Downlaod them from [data](https://drive.google.com/file/d/1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im/view). Then, please unzip and save them (data) to the corresopinding dir.

```bash
unzip data.zip -d data/finetune
```

### Excute Fine-tuning
- After pre-training the Coke model, move pytorch_model.bin to the corresponding dir
DKPLM/data/DKPLM_BERTbase_2layer DKPLM/data/DKPLM_RoBERTabase_2layer

```bash
export BACKBONE=bert-base-uncased
export HOP=2

mv outputs/pretrain_coke-$BACKBONE-$HOP/pytorch_model.bin ../checkpoint/coke-$BACKBONE/pytorch_model.bin
```

#### FewRel/Figer/Open Entity/TACRED
```
bash run_finetune.sh
```


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
