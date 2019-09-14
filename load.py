import gensim
import torch
model = gensim.models.KeyedVectors.load_word2vec_format('/home/zhangjinjie/knowledge-driven-dialogue-master2/knowledge-driven-dialogue-master/generative_pt/sgns.baidubaike.bigram-char',binary=False)
print(model.index2word)
voc = {'的':0, '在':1, '和':2, '是':3, '了':4,'爱':5}
weight = torch.zeros(6,300)
for word in voc.keys():
    index = voc[word]
    if word in model.index2word:
        weight[index,:] = torch.from_numpy(model[word])
embedding = torch.nn.Embedding.from_pretrained(weight)
#self.embedding.weight.requires_grad = False
print()
