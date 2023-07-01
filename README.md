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
#### v1.0
- [x] CokeBert
- [x] CokeRoberta
#### v2.0
- [x] CokeBert
- [x] CokeRoberta (will release soon)  

## Reqirements
- python>=3.8
- torch>=1.9.0
- transformers>=4.10

Please install all required packages by running

```bash
pip install -r requirements.txt
```

## Data preparation
#### Pre-training Data

Skip this section if you do not want to do pre-training on your own.

Go to the folder for the latest version. Choose a backbone model, e.g. `bert-base-uncased`
```bash
cd CokeBert-2.0-latest
```

Pleas follow the [ERNIE](https://github.com/thunlp/ERNIE "ERNIE") pipline to pre-process your data, using the corresponding tokenizer of the backbone model. The outputs are `merbe.bin` and `merge.idx`. 


After pre-process Pre-trained data, move them to the corresopinding directory.

```bash
export BACKBONE=bert-base-uncased
export HOP=2

mkdir data/pretrain/$BACKBONE

mv merge.bin data/pretrain/$BACKBONE
mv mergr.idx data/pretrain/$BACKBONE
```

Download the backbone model checkpoint from [Huggingface](https://huggingface.co/models), and move it to the corresponding checkpoint folder for pre-training. Note do not download the `config.json` for the backbone model, since we will be using the config of `coke`.

```bash
wget https://huggingface.co/$BACKBONE/resolve/main/vocab.txt -O checkpoint/coke-$BACKBONE/vocab.txt
wget https://huggingface.co/$BACKBONE/resolve/main/pytorch_model.bin -O checkpoint/coke-$BACKBONE/pytorch_model.bin
```

Download the Knowledge Embedding (including entity and relation to id information) and knowledge graph neighbor information from [here](https://drive.google.com/drive/folders/116FG9U-U4r674dgfBBL3qMceGXTTcaxc?usp=sharing). Put them in the `data/pretrain` folder and unzip them.

```bash
cd data/pretrain

tar zxvf kg_embed.tar.gz
tar zxvf kg_neighbor.tar.gz

cd ../..
```

(*Optional*) Generate Knowledge Graph Neighbors. We have provided this data. If you want to change the max number of neighbors, you can run this code to get the new `kg_neighbor` data
```bash
cd data/pretrain
python3 preprocess_n.py
```


#### Fine-tuning data

As most datasets except FewRel do not have entity annotations, we use the annotated dataset from ERNIE. Downlaod them from [data](https://drive.google.com/file/d/1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im/view). Then, please unzip and save them (data) to the corresopinding dir.

```bash
unzip data.zip -d data/finetune
```

## Use pre-trained Coke checkpoint

You can download the pre-trained Coke checkpoints from [here](https://huggingface.co/yushengsu/CokeBERT) and start using it in python. For example, the following code loads a 2-hop Coke `Bert-base` model (also in `CokeBert-2.0-latest/examples/run_finetune.py`)

```python
from coke import CokeBertModel

model = CokeBertModel.from_pretrained('yushengsu/coke-bert-base-uncased-2hop')

# Do something with the model
```

## Pre-train from scratch
If you want to run pre-training on your own, please first prepare the pre-training data. Then run the following commands (also in `CokeBert-2.0-latest/examples/run_pretrain.sh`)

```bash
cd CokeBert-2.0-latest/examples
export BACKBONE=bert-base-uncased
export HOP=2
export PYTHONPATH=../src:$PYTHONPATH

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
If you want to fine-tune Coke model on downstream tasks, please first prepare the fine-tuning data. Then move the pre-trained Coke model checkpoint file `pytorch_model.bin` to the corresponding dir, such as `DKPLM/data/DKPLM_BERTbase_2layer` for 2-hop `Bert-base` model and `DKPLM/data/DKPLM_RoBERTabase_2layer` for 2-hop `Roberta-base` model.

```bash
mv outputs/pretrain_coke-$BACKBONE-$HOP/pytorch_model.bin ../checkpoint/coke-$BACKBONE/pytorch_model.bin
```

Then start fine-tuning by running the following commands (also in `CokeBert-2.0-latest/examples/run_finetune.sh`)

```bash
cd CokeBert-2.0-latest/examples
export BACKBONE=bert-base-uncased
export HOP=2
export PYTHONPATH=../src:$PYTHONPATH

python3 run_finetune.py \
            --output_dir outputs \
            --do_train \
            --do_lower_case \
            --data_dir ../data/finetune/fewrel/ \
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

Currently we support the following datasets
- FewRel
- Figer
- Open Entity
- TACRED


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
