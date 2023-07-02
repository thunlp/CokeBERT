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


## Code Version
- [CokeBert-1.0](https://github.com/thunlp/CokeBERT/tree/main/CokeBert-1.0) provides the original codes and details to reproduce the results in the paper.
- [CokeBert-2.0-latest](https://github.com/thunlp/CokeBERT) refactors the CokeBert-1.0 and provides more user-friendly codes for users. In this `README.md`, we mainly demostrate the usage of the `CokeBert-2.0-latest`.  


<!--
#### v1.0
- [x] CokeBert
- [x] CokeRoberta
#### v2.0
- [x] CokeBert
- [x] CokeRoberta (will release soon)  
-->

## Reqirements
- python==3.8
<!--
- python>=3.8
- torch>=1.9.0
- transformers>=4.10
-->

Please install all required packages by running

```bash
bash requirements.sh
```



## Pre-training
If you want to use our pre-trained Coke models directly, you can ignore this section and skip to fine-tuning part.

### Data preparation
Go to `CokeBert-2.0-latest`
```bash
cd CokeBert-2.0-latest
```

#### Step1
Please follow the [ERNIE](https://github.com/thunlp/ERNIE "ERNIE") pipline to pre-process your pre-training data. Note that you need to dicide the backbone model and utilize its corresponding tokenizer to process the data. Coke framework supports two series of models (`BERT` and `RoBERTa`) currently. Then, you will obtain `merbe.bin` and `merge.idx` and move them to the following directories.

```bash
export BACKBONE=BACKBONE
# For example: export BACKBONE=bert-base-uncased
export HOP=2

mkdir data/pretrain/$BACKBONE

mv merge.bin data/pretrain/$BACKBONE
mv mergr.idx data/pretrain/$BACKBONE
```

#### Step2
Download the backbone model checkpoints from [Huggingface](https://huggingface.co/models), and move them to the corresponding checkpoint folder for pre-training. Note you do not download the `config.json`, since we create new config for `coke`.
```bash
$BACKBONE=BACKBONE
# For example: BACKBONE can be `bert-base-uncased`, `roberta-base`

wget https://huggingface.co/$BACKBONE/resolve/main/vocab.txt -O checkpoint/coke-$BACKBONE/vocab.txt
wget https://huggingface.co/$BACKBONE/resolve/main/pytorch_model.bin -O checkpoint/coke-$BACKBONE/pytorch_model.bin

mv vocab.txt $BACKBONE/
mv pytorch_model.bin $BACKBONE/
mv $BACKBONE checkpoint/
```

#### Step3
Download the Knowledge Embedding (including entity and relation to id information) and knowledge graph neighbor information from [here1](https://drive.google.com/drive/folders/116FG9U-U4r674dgfBBL3qMceGXTTcaxc?usp=sharing) or [here2](https://cloud.tsinghua.edu.cn/d/dd92eb793c224cea8ec9/). Move them to `data/pretrain` folder and unzip them.

```bash
cd data/pretrain

tar zxvf kg_embed.tar.gz
rm -rf kg_embed.tar.gz

tar zxvf kg_neighbor.tar.gz
rm -rf kg_neighbor

cd ../..
```

(*Optional*) If you want to generate knowledge graph neighbors by yourself, you can 
run this code to get the new `kg_neighbor` data.
```bash
cd data/pretrain
python3 preprocess_n.py
```


### Start to Train
Go to examples and run the `run_pretrain.sh`.
```bash
cd example
bash run_pretrain.sh
```
You can assign `BACKBONE` (backbone models) and `HOP` (the number of hop) in `run_pretrain.sh`.
```bash
export BACKBONE=BACKBONE
# BACKBONE can be `bert-base-uncased`, `roberta-base`, etc
export HOP=2
export PYTHONPATH=../src:$PYTHONPATH

rm outputs/pretrain_coke-$BACKBONE-$HOP/*

python run_pretrain.py \
            --output_dir outputs \
            --data_dir ../data/pretrain \
            --backbone $BACKBONE \
            --neighbor_hop $HOP \
            --do_train \
            --max_seq_length 256 \
            --K_V_dim 100 \
            --Q_dim 768 \
            --train_batch_size 32 \
            --self_att
```
It will write log and checkpoint to `./outputs`. Check `CokeBert-2.0-latest/src/coke/training_args.py` for more arguments.






## Fine-tuning


### Data preparation
We download the fine-tuning datasets and the coresponding annotations from [here1](https://drive.google.com/file/d/1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im/view) or [here2](https://cloud.tsinghua.edu.cn/f/3036fa28168c4fb7a320/?dl=1). Then, please unzip and save them to the corresopinding dir.

```bash
cd CokeBert-2.0-latest/data
wget https://cloud.tsinghua.edu.cn/f/3036fa28168c4fb7a320/?dl=1 
mv 'index.html?dl=1' data.zip
tar -xvf data.zip finetune
```


### Start to Train
(*Optiona1: Load from Huggieface*) You can load the pre-trained Coke checkpoints from [here](https://huggingface.co/yushengsu/CokeBERT) and start using it in python and start to fine-tune. For example, the following code demostrates how to load a 2-hop Coke `bert-base` model.

```python
from coke import CokeBertModel

model = CokeBertModel.from_pretrained('yushengsu/coke-bert-base-uncased-2hop')

# You can use this model to start fine-tune.
```


(*Option2: Load from the local*) You can also downlaod the pre-trained Coke checkpoints from [here](https://huggingface.co/yushengsu/CokeBERT) and run the following script to fine-tune. Note that you need to  move the pre-trained Coke model checkpoints `pytorch_model.bin` to the corresponding dir, such as `DKPLM/data/DKPLM_BERTbase_2layer` for 2-hop `bert-base-uncased` model and `DKPLM/data/DKPLM_RoBERTabase_2layer` for 2-hop `roberta-base` model.

```bash
# $BACKBONE=BACKBONE (`bert-base-uncased`, `roberta-base`, etc.)
# $HOP=HOP (1 or 2)

mv outputs/pretrain_coke-$BACKBONE-$HOP/pytorch_model.bin ../checkpoint/coke-$BACKBONE/pytorch_model.bin
```

Then start fine-tuning by running the following commands (Refer to `CokeBert-2.0-latest/examples/run_finetune.sh`)

<!--
FewRel
Figer
Open Entity
TACRED
-->

```bash
cd CokeBert-2.0-latest/examples
export BACKBONE=$BACKBONE
# BACKBONE can be `bert-base-uncased`, `roberta-base`, etc.
export HOP=2
export PYTHONPATH=../src:$PYTHONPATH
DATASET=DATASET
# $DATASET can be `FIGER`, `OpenEntity`, `fewrel`, `tacred`

python3 run_finetune.py \
            --output_dir outputs \
            --do_train \
            --do_lower_case \
            --data_dir ../data/finetune/$DATASET/ \
            --backbone $BACKBONE \
            --neighbor_hop $HOP \
            --max_seq_length 256 \
            --train_batch_size 64 \
            --learning_rate 2e-5 \
            --num_train_epochs 16 \
            --loss_scale 128 \
            --K_V_dim 100 \
            --Q_dim 768 \
            --self_att
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
