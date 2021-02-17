import torch
from collections import defaultdict
import pickle
from tqdm import tqdm
import sys

#max_neighbor = 800
max_neighbor = 20
_in = True
max_entity_id = 5040986
file = sys.argv[1].split("_")[0]
filename = sys.argv[1]




#######################################
#######################################
#######################################
#######################################


#load entity pairs -->undirected
#e_r_e = defaultdict(lambda : defaultdict(set))
e1_e2 = defaultdict(list)
e1_r = defaultdict(list)
e1_outORin = defaultdict(list)

with open("../../data/{}/{}_graph.csv".format(file,filename)) as fp:
#with open("../../data/tacred_graph_tr_eval.csv") as fp:
#with open("../../data/tacred_graph_te.csv") as fp:
#with open("../../kg_embed/train2id.txt") as fp:
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


'''
with open('load_data_small/e1_e2_tacred_tr_eval.pkl', 'wb') as f:
    pickle.dump(e1_e2, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_small/e1_r_tacred_tr_eval.pkl', 'wb') as f:
    pickle.dump(e1_r, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_small/e1_outORin_tacred_tr_eval.pkl', 'wb') as f:
    pickle.dump(e1_outORin, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Save triple_list DONE!!")
'''




#######################################
#######################################
#######################################



###Load pickle: dict() ==> produce: e+r
#with open('load_data/e1_e2.pkl', 'rb') as f:
#    e1_e2 = pickle.load(f)
#with open('load_data/e1_r.pkl', 'rb') as f:
#    e1_r = pickle.load(f)
#with open('load_data/e1_outORin.pkl', 'rb') as f:
#    e1_outORin = pickle.load(f)



#save to e: list
#max_neighbor = 2000
e1_e2_list_2D = list()
e1_r_list_2D = list()
e1_outORin_list_2D = list()
#for i,e in enumerate(e1_e2):
#for i,e in enumerate(tqdm(e1_e2)):
for i in tqdm(range(max_entity_id+1)):
    if len(e1_e2[i]) >= max_neighbor:
        e1_e2_list_2D.append(e1_e2[i][:max_neighbor])
    else:
        e1_e2_list_2D.append(e1_e2[i]+[0]*(max_neighbor-len(e1_e2[i])))


    if len(e1_r[i]) >= max_neighbor:
        e1_r_list_2D.append(e1_r[i][:max_neighbor])
    else:
        e1_r_list_2D.append(e1_r[i]+[0]*(max_neighbor-len(e1_r[i])))


    if len(e1_outORin[i]) >= max_neighbor:
        e1_outORin_list_2D.append(e1_outORin[i][:max_neighbor])
    else:
        e1_outORin_list_2D.append(e1_outORin[i]+[0]*(max_neighbor-len(e1_outORin[i])))



e1_e2_list_2D_Tensor = torch.LongTensor(e1_e2_list_2D)
del e1_e2
e1_r_list_2D_Tensor = torch.LongTensor(e1_r_list_2D)
del e1_r
e1_outORin_list_2D_Tensor = torch.FloatTensor(e1_outORin_list_2D)
del e1_outORin


with open('load_data_small/e1_e2_list_2D_Tensor_{}.pkl'.format(filename), 'wb') as f:
    pickle.dump(e1_e2_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_small/e1_r_list_2D_Tensor_{}.pkl'.format(filename), 'wb') as f:
    pickle.dump(e1_r_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('load_data_small/e1_outORin_list_2D_Tensor_{}.pkl'.format(filename), 'wb') as f:
    pickle.dump(e1_outORin_list_2D_Tensor, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Save Triple Tensor DONE!!")


exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()
exit()


#######################################
#######################################
#######################################
