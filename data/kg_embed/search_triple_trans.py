from collections import defaultdict
import json


pid_rel=dict()
with open('pid2rel_all.json') as json_file:
    data = json.load(json_file)
    for pid in data:
        pid_rel[pid] = data[pid][0]



id_q=dict()
with open('entity2id.txt') as f:
    for line in f:
        #q, id = line.strip().split('\t')
        line = line.strip().split('\t')
        if len(line) ==1:
            continue
        q, id = line
        id_q[id] = q


q_ent=dict()
with open("entity_map.txt") as f:
    for line in f:
        line = line.strip().split('\t')
        if len(line) == 1:
            continue
        ent, q = line
        q_ent[q] = ent

id_p__rel = dict()
with open("relation2id.txt") as f:
    for line in f:
        #p, id = line.strip().split('\t')
        line = line.strip().split('\t')
        if len(line) ==1:
            continue
        p, id = line
        id_p__rel[id] = p


all_triple = defaultdict(lambda : defaultdict(list))
with open("train2id.txt") as f:
    for line in f:
        #e_1,e_2,r = line.strip().split()
        line = line.strip().split()
        if len(line)==1:
            continue
        e_1, e_2, r = line
        all_triple[e_1][e_2].append(r)

#Bill Gate
search_id = 250297

#Mark Zuckerberg
#search_id = 1006293

search_id = str(search_id-1)
print("search_id:",search_id)
print("Ent:",q_ent[id_q[search_id]])
#e_ = id_q[search_id]
#print(e_)
#e__ = q_ent[e_]
#print(e__)
print("=========================")
e_2_s = all_triple[search_id]
for e_2_id in e_2_s:
    try:
        e_1 = q_ent[id_q[search_id]]
        e_2 = q_ent[id_q[e_2_id]]
        r = all_triple[search_id][e_2_id]
    except:
        print("have no rel or ent")
        continue
    if len(r)!=1:
        exit()
    r = pid_rel[id_p__rel[all_triple[search_id][e_2_id][0]]]
    print(e_1,",",e_2,",",r)


