import torch
import pickle
from tqdm import tqdm
from collections import defaultdict


max_neighbor = 20
_in = True
max_entity_id = 5040986

def load_knowledge():
    vecs = []
    vecs.append([0]*100) # CLS
    with open("./kg_embed/entity2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs

    vecs = []
    vecs.append([0]*100) # CLS
    with open("./kg_embed/relation2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    del vecs
    embed_ent = torch.nn.Embedding.from_pretrained(embed_ent)
    embed_r = torch.nn.Embedding.from_pretrained(embed_r)

    return embed_ent, embed_r

e1_e2 = defaultdict(list)
e1_r = defaultdict(list)
e1_outORin = defaultdict(list)
with open("./kg_embed/train2id.txt") as fp:

    for e in range(0,max_entity_id+1):
        e1_e2[e].append(e)
        e1_r[e].append(0)
        e1_outORin[e].append(0)

    for i,line in enumerate(fp):
        info = line.strip().split()
        if len(info) < 2:
            continue
        else:
            e1 = int(info[0])+1
            e2 = int(info[1])+1
            r = int(info[2])+1

            e1_e2[e1].append(e2)
            e1_r[e1].append(r)
            e1_outORin[e1].append(-1)

            if _in:
                e1_e2[e2].append(e1)
                e1_r[e2].append(r)
                e1_outORin[e2].append(1)


print("Save Triple Tensor DONE!!")

with open('./kg_neighbor/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    e1_e2_list_2D_Tensor = pickle.load(f)
with open('./kg_neighbor/e1_r_list_2D_Tensor.pkl', 'rb') as f:
    e1_r_list_2D_Tensor = pickle.load(f)
with open('./kg_neighbor/e1_outORin_list_2D_Tensor.pkl', 'rb') as f:
    e1_outORin_list_2D_Tensor = pickle.load(f)

embed_ent, embed_r = load_knowledge()

e1_e2_emb = embed_ent(e1_e2_list_2D_Tensor)

print(e1_e2_emb.shape)
del e1_e2_list_2D_Tensor

e1_r_emb = embed_r(e1_r_list_2D_Tensor)
del e1_r_list_2D_Tensor

outORin = e1_outORin_list_2D_Tensor.unsqueeze(2)
del e1_outORin_list_2D_Tensor

e_ = e1_e2_emb + outORin*e1_r_emb

with open('./kg_neighbor/e_.pkl', 'wb') as f:
    pickle.dump(e_, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Save e_ DONE!!")
