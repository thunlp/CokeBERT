import pickle

with open('load_data/e1_e2_list_2D_Tensor.pkl', 'rb') as f:
    e1_e2_list_2D_Tensor = pickle.load(f)

print(e1_e2_list_2D_Tensor)


'''
a={1:[1],2:[2],3:[3]}
print(a)

b=list()
for i in range(len(a)):
    print(a[i+1])
    b.append(a[i+1]+[0]*3)
print(a)
print(b)
'''
