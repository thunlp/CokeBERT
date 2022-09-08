import os
import pickle
import torch


def load_kg_embedding(data_dir):
    '''Load KG embedding'''

    # Entity
    vecs = []
    vecs.append([0]*100) # CLS
    with open(os.path.join(data_dir, 'kg_embed/entity2vec.vec'), 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_ent = torch.FloatTensor(vecs)
    del vecs

    # Relation
    vecs = []
    vecs.append([0]*100) # CLS
    with open(os.path.join(data_dir, 'kg_embed/relation2vec.vec'), 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)
    embed_r = torch.FloatTensor(vecs)
    del vecs

    return embed_ent, embed_r

def load_ent_emb_dynamic(data_dir):
    with open(os.path.join(data_dir, 'kg_neighbor/e1_e2.pkl'), 'rb') as f:
        ent_neighbor = pickle.load(f)

    with open(os.path.join(data_dir, 'kg_neighbor/e1_r.pkl'), 'rb') as f:
        ent_r = pickle.load(f)

    with open(os.path.join(data_dir, 'kg_neighbor/e1_outORin.pkl'), 'rb') as f:
        ent_outORin = pickle.load(f)

    return ent_neighbor, ent_r, ent_outORin


def load_ent_emb_static(data_dir):
    with open(os.path.join(data_dir, 'kg_neighbor/e1_e2_list_2D_Tensor.pkl'), 'rb') as f:
        ent_neighbor = pickle.load(f)

    with open(os.path.join(data_dir, 'kg_neighbor/e1_r_list_2D_Tensor.pkl'), 'rb') as f:
        ent_r = pickle.load(f)

    with open(os.path.join(data_dir, 'kg_neighbor/e1_outORin_list_2D_Tensor.pkl'), 'rb') as f:
        ent_outORin = pickle.load(f)

    return ent_neighbor, ent_r, ent_outORin


def k_v(input_ent, ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r, hop=2):
    assert hop in [1, 2], '"hop" should be chosen from [1, 2].'
    
    # Neighbor
    input_ent_neighbor = torch.index_select(ent_neighbor, 0, input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()

    # create input_ent_neighbor_1
    input_ent_neighbor_emb_1 = torch.index_select(embed_ent,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])) #
    input_ent_neighbor_emb_1 = input_ent_neighbor_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],embed_ent.shape[-1])

    # create input_ent_r_1:
    print(ent_r.shape, input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]).max())
    input_ent_r_emb_1 = torch.index_select(ent_r, 0, input_ent.reshape(input_ent.shape[0]*input_ent.shape[1])).long()
    input_ent_r_emb_1 = torch.index_select(embed_r, 0, input_ent_r_emb_1.reshape(input_ent_r_emb_1.shape[0]*input_ent_r_emb_1.shape[1])) #
    input_ent_r_emb_1 = input_ent_r_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],embed_r.shape[-1])

    # create outORin_1:
    input_ent_outORin_emb_1 = torch.index_select(ent_outORin,0,input_ent.reshape(input_ent.shape[0]*input_ent.shape[1]))
    input_ent_outORin_emb_1 = input_ent_outORin_emb_1.reshape(input_ent.shape[0],input_ent.shape[1],input_ent_outORin_emb_1.shape[1])
    input_ent_outORin_emb_1 = input_ent_outORin_emb_1.unsqueeze(3)

    # create input_ent_neighbor_2
    input_ent_neighbor_2 = torch.index_select(ent_neighbor,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
    input_ent_neighbor_emb_2 = torch.index_select(embed_ent,0,input_ent_neighbor_2.reshape(input_ent_neighbor_2.shape[0]*input_ent_neighbor_2.shape[1])) #
    input_ent_neighbor_emb_2 = input_ent_neighbor_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_neighbor.shape[1],ent_neighbor.shape[1],100)

    # create input_ent_r_2:
    input_ent_r_2 = torch.index_select(ent_r,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1])).long()
    input_ent_r_emb_2 = torch.index_select(embed_r,0,input_ent_r_2.reshape(input_ent_r_2.shape[0]*input_ent_r_2.shape[1])) #
    input_ent_r_emb_2 = input_ent_r_emb_2.reshape(input_ent.shape[0],input_ent.shape[1],ent_r.shape[1],ent_neighbor.shape[1],100)

    # create outORin_2:
    input_ent_outORin_emb_2 = torch.index_select(ent_outORin,0,input_ent_neighbor.reshape(input_ent_neighbor.shape[0]*input_ent_neighbor.shape[1]))
    input_ent_outORin_emb_2 = input_ent_outORin_emb_2.reshape(input_ent_r_emb_2.shape[0],input_ent_r_emb_2.shape[1],input_ent_r_emb_2.shape[2],input_ent_r_emb_2.shape[3])
    input_ent_outORin_emb_2 = input_ent_outORin_emb_2.unsqueeze(4)

    k_1 = input_ent_outORin_emb_1.cuda() * input_ent_r_emb_1.cuda()
    v_1 = input_ent_neighbor_emb_1.cuda() + k_1
    k_2 = input_ent_outORin_emb_2.cuda() * input_ent_r_emb_2.cuda()
    v_2 = input_ent_neighbor_emb_2.cuda() + k_2

    if hop == 1:
        output = k_1, v_1, None, None
    elif hop == 2:
        output = k_1, v_1, k_2, v_2
    
    return output


def load_k_v_queryR_small(input_ent, candidate, ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r, hop=2):
    assert hop in [1, 2], '"hop" should be chosen from [1, 2].'

    input_ent = input_ent.cpu()
    candidate = candidate.cpu()

    ent_pos_s = torch.nonzero(input_ent)

    max_entity = 0
    value = 0
    idx_1 = 0
    last_part = 0
    for idx_2, x in enumerate(ent_pos_s):
        if int(x[0]) != value:
            max_entity = max(idx_2-idx_1,max_entity)
            idx_1 = idx_2
            value = int(x[0])
            last_part = 1
        else:
            last_part += 1
    max_entity = max(last_part, max_entity)

    new_input_ent = list()
    cand_pos_tensor = torch.LongTensor(input_ent.shape[0],candidate.shape[1]) # V
    for i_th, ten in enumerate(input_ent):
        ten_ent = ten[ten!=0]
        ten_ent_uniqe = ten_ent.unique() #v
        cand_pos_tensor[i_th][:] = torch.LongTensor([int(ent in ten_ent_uniqe) for ent in candidate[0]]) # V

        new_input_ent.append( torch.cat( (ten_ent,( torch.LongTensor( [0]*(max_entity-ten_ent.shape[0]) ) ) ) ) )
    input_ent = torch.stack(new_input_ent)

    k_1, v_1, k_2, v_2 = k_v(input_ent, ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r, hop)
    k_cand_1, v_cand_1, k_cand_2, v_cand_2 = k_v(candidate, ent_neighbor, ent_r, ent_outORin, embed_ent, embed_r, hop)

    return k_1, v_1, k_2, v_2, k_cand_1, v_cand_1, k_cand_2, v_cand_2, cand_pos_tensor
