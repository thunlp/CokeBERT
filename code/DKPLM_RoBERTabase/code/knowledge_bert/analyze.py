import torch
from collections import defaultdict
import pickle
from tqdm import tqdm
import csv

#max_neighbor = 800
#max_neighbor = 300
max_neighbor = 100
_in = True
max_entity_id = 5040986
top_n = 10

def load_knowledge():
    #load KG emb
    vecs = []
    vecs.append([0]*100) # CLS
    with open("../../kg_embed/entity2vec.vec", 'r') as fin:
    #with open("../../kg_embed/entity2vec.del", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs


    #load relation emb
    vecs = []
    vecs.append([0]*100) # CLS
    with open("../../kg_embed/relation2vec.vec", 'r') as fin:
    #with open("../../kg_embed/relation2vec.del", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    del vecs


    embed_ent = torch.nn.Embedding.from_pretrained(embed_ent)
    embed_r = torch.nn.Embedding.from_pretrained(embed_r)

    #return embed_ent, embed_r, e1_e2_r_outORin
    print("Load Emb DONE!!")
    return embed_ent, embed_r



#######################################
#######################################
#######################################
#######################################


#load entity pairs -->undirected
#e_r_e = defaultdict(lambda : defaultdict(set))
#e1_e2_out = defaultdict(list)
#e1_r_out = defaultdict(list)
#e1_out = defaultdict(list)
#e1_e2_in = defaultdict(list)
#e1_r_in = defaultdict(list)
#e1_in = defaultdict(list)
e1_e2 = defaultdict(list)
e1_r = defaultdict(list)
e1_outORin = defaultdict(list)
#e1_outORin = defaultdict(list)
with open("../../kg_embed/train2id.txt") as fp:
#with open("../../kg_embed/train2id.del") as fp:

    #self
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

            #######################################
            #######################################

            #e1_e2
            #e1_e2[e1].append(e1)
            e1_e2[e1].append(e2)
            #e1_r
            #e1_r[e1].append(0)
            e1_r[e1].append(r)
            #e1_outORin
            #e1_outORin[e1].append(0)
            e1_outORin[e1].append(-1)

            if _in:
                e1_e2[e2].append(e1)
                e1_r[e2].append(r)
                e1_outORin[e2].append(1)


##Caculate: Node Neighbors similiarty
embed_ent, embed_r = load_knowledge()
#for id in range(max_entity_id+1):
#with open("report.csv", "wb") as csv_file:
    #writer = csv.writer(csv_file)

mean_sort_list=list()
for id in tqdm(range(max_entity_id+1)):
#for id in tqdm(range(10)):
    e1_e2_2D_Tensor = torch.LongTensor(e1_e2[id])
    e1_r_2D_Tensor= torch.LongTensor(e1_r[id])
    e1_outORin_2D_Tensor = torch.FloatTensor(e1_outORin[id])
    #print("===========")
    e1 = embed_ent(e1_e2_2D_Tensor)
    r = embed_r(e1_r_2D_Tensor)
    #if id != 0:
    outORin = e1_outORin_2D_Tensor.unsqueeze(1)
    e1_neighbors = e1+r*outORin

    e1 = embed_ent(torch.LongTensor([id]*e1_neighbors.shape[0]))
    cos_sim = torch.cosine_similarity(e1,e1_neighbors,dim=1)

    if cos_sim.shape[0] > top_n:
        cos_sim_sort = cos_sim.sort(dim=0).values[:top_n] #small --> big
    else:
        cos_sim_sort = cos_sim.sort(dim=0).values #small --> big
    cos_sim_mean = cos_sim.mean()

    #mean_sort_list.append([cos_sim_mean,cos_sim_sort])

    print(i)
    print(cos_sim_mean)
    print(cos_sim_sort)
    print("================")
    #writer.writerow(str([str(cos_sim_mean),str(cos_sim_sort)]))
    #writer.writerow(str(cos_sim_mean))
    #writer.writerow(str(cos_sim_sort))
    #writer.writerow("================\n")



