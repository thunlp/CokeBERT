import torch
from collections import defaultdict
import pickle
from tqdm import tqdm

#max_neighbor = 800
#max_neighbor = 300
#max_neighbor = 100
max_neighbor = 4
_in = True
#max_entity_id = 5040986
max_entity_id = 9

def load_knowledge():
    #load KG emb
    vecs = []
    vecs.append([0]*100) # CLS
    #with open("../../kg_embed/entity2vec.vec", 'r') as fin:
    with open("../../kg_embed/entity2vec.del", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs


    #load relation emb
    vecs = []
    vecs.append([0]*100) # CLS
    #with open("../../kg_embed/relation2vec.vec", 'r') as fin:
    with open("../../kg_embed/relation2vec.del", 'r') as fin:
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
e1_e2_out = defaultdict(list)
e1_r_out = defaultdict(list)
e1_out = defaultdict(list)
e1_e2_in = defaultdict(list)
e1_r_in = defaultdict(list)
e1_in = defaultdict(list)
#e1_outORin = defaultdict(list)
#with open("../../kg_embed/train2id.txt") as fp:
with open("../../kg_embed/train2id.del") as fp:

    #self
    for e in range(0,max_entity_id+1):
        e1_e2_out[e].append(e)
        e1_r_out[e].append(0)
        e1_out[e].append(0)


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
            e1_e2_out[e1].append(e2)
            #e1_r
            #e1_r[e1].append(0)
            e1_r_out[e1].append(r)
            #e1_outORin
            #e1_outORin[e1].append(0)
            e1_out[e1].append(-1)

            if _in:
                e1_e2_in[e2].append(e1)
                e1_r_in[e2].append(r)
                e1_in[e2].append(1)


#print(e1_e2_in)
#print(e1_r_in)
#print(e1_out)


e1_e2 = defaultdict(list)
e1_r = defaultdict(list)
e1_outORin = defaultdict(list)
#c_ = 0
for i in tqdm(range(max_entity_id+1)):
    #e1_e2
    if len(e1_e2_out[i]) < max_neighbor:
        #c_ +=1
        #print(1)
        if len(e1_e2_in[i]) < max_neighbor-len(e1_e2_out[i]):
            e1_e2[i].append(e1_e2_out[i]+e1_e2_in[i]+[0]*(max_neighbor-len(e1_e2_out[i])-len(e1_e2_in[i])))
            #print("---------")
            #print(e1_e2_in[i])
            #print(len(e1_e2_in[i]))
            #print(e1_e2_out[i])
            #print(len(e1_e2_out[i]))
            #print([0]*(max_neighbor-len(e1_e2_out)-len(e1_e2_in)))
            #print(e1_e2_out[i]+e1_e2_in[i]+[0]*(max_neighbor-len(e1_e2_out[i])-len(e1_e2_in[i])))
            #print("---------")
            #print(2)
        else:
            e1_e2[i].append(e1_e2_out[i]+e1_e2_in[i][:max_neighbor-len(e1_e2_out[i])])
            #print(3)
    else:
        e1_e2[i].append(e1_e2_out[i][:max_neighbor])
        #print(4)
    #print("---")

    #e_r
    if len(e1_r_out[i]) < max_neighbor:
        if len(e1_r_in[i]) < max_neighbor-len(e1_r_out[i]):
            e1_r[i].append(e1_r_out[i]+e1_r_in[i]+[0]*(max_neighbor-len(e1_r_out[i])-len(e1_r_in[i])))
        else:
            e1_r[i].append(e1_r_out[i]+e1_r_in[i][:max_neighbor-len(e1_r_out[i])])
    else:
        e1_r[i].append(e1_r_out[i][:max_neighbor])

    #e_outORout
    if len(e1_out[i]) < max_neighbor:
        if len(e1_in[i]) < max_neighbor-len(e1_out[i]):
            e1_outORin[i].append(e1_out[i]+e1_in[i]+[0]*(max_neighbor-len(e1_out[i])-len(e1_in[i])))
        else:
            e1_outORin[i].append(e1_out[i]+e1_in[i][:max_neighbor-len(e1_out[i])])
    else:
        e1_outORin[i].append(e1_out[i][:max_neighbor])


    if len(e1_e2[i][0])!=max_neighbor:
        print(e1_e2[i][0])
        print(len(e1_e2[i][0]))
    #if len(e1_r[i])<100:
    #    print(e1_r[i])
    #if len(e1_outORin[i])<100:
    #    print(e1_outORin[i])
        print("i:",i)
        exit()
    #print("==============")

#print(c_)
#print("{}%".format(100*c_/max_entity_id))
#exit()

with open('load_data_test/e1_e2.pkl', 'wb') as f:
    pickle.dump(e1_e2, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_test/e1_r.pkl', 'wb') as f:
    pickle.dump(e1_r, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_test/e1_outORin.pkl', 'wb') as f:
    pickle.dump(e1_outORin, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Save triple_list DONE!!")





#save to e: list
#max_entity_length = len(e1_e2)
#max_neighbor = 2000
e1_e2_list_2D = list()
e1_r_list_2D = list()
e1_outORin_list_2D = list()
#for i,e in enumerate(e1_e2):
for i,e in enumerate(tqdm(e1_e2)):
    '''
    if len(e1_e2[e]) > max_neighbor:
        e1_e2_list_2D.append(e1_e2[e][:max_neighbor])
    else:
        e1_e2_list_2D.append(e1_e2[e]+[0]*(max_neighbor-len(e1_e2[e])))


    if len(e1_r[e]) > max_neighbor:
        e1_r_list_2D.append(e1_r[e][:max_neighbor])
    else:
        e1_r_list_2D.append(e1_r[e]+[0]*(max_neighbor-len(e1_r[e])))


    if len(e1_outORin[e]) > max_neighbor:
        e1_outORin_list_2D.append(e1_outORin[e][:max_neighbor])
    else:
        e1_outORin_list_2D.append(e1_outORin[e]+[0]*(max_neighbor-len(e1_outORin[e])))
    '''

    e1_e2_list_2D.append(e1_e2[i][0])
    #print(e1_e2_list_2D)
    #exit()
    e1_r_list_2D.append(e1_r[i][0])
    e1_outORin_list_2D.append(e1_outORin[i][0])




e1_e2_list_2D_Tensor = torch.LongTensor(e1_e2_list_2D)
del e1_e2
e1_r_list_2D_Tensor = torch.LongTensor(e1_r_list_2D)
del e1_r
e1_outORin_list_2D_Tensor = torch.FloatTensor(e1_outORin_list_2D)
del e1_outORin


with open('load_data_test/e1_e2_list_2D_Tensor.pkl', 'wb') as f:
    pickle.dump(e1_e2_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_test/e1_r_list_2D_Tensor.pkl', 'wb') as f:
    pickle.dump(e1_r_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_test/e1_outORin_list_2D_Tensor.pkl', 'wb') as f:
    pickle.dump(e1_outORin_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Save Triple Tensor DONE!!")

