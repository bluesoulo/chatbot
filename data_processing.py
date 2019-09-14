import random
import torch
import pickle
import gensim
import jieba
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
def get_pairs():
    f1 = open('./data/train.enc', 'r',encoding='UTF-8-sig')
    f2 = open('./data/train.dec','r',encoding='UTF-8-sig')
    question = [line.strip('\n') for line in f1]
    answer = [line.strip('\n') for line in f2]
    if len(question) < len(answer):
        answer = answer[:len(question)]
    else:
        question = question[:len(answer)]
    '''
    pairs = []
    i, j, k = 0, 0, 0
    for x, y in zip(question, answer):
        if len(x) < 250 and len(y) < 70 :
            i += 1
        if len(x) <250 :
            j = j+1
        if len(y) < 70:
            k += 1
        pairs.append([x,y])
    '''
    pairs = []
    for x, y in zip(question, answer):
        ls_x = '/'.join(jieba.cut(x, cut_all=False)).split('/')
        ls_y = '/'.join(jieba.cut(y, cut_all=False)).split('/')
        if len(ls_x) < 150 and len(ls_y) < 60 :
             pairs.append([ls_x,ls_y])
    return pairs
def get_directionary(min_count = 3 ,max_length = 10000):
    voc,voc_final = {},{}
    f= open('./data/train.enc','r',encoding='utf-8')
    for line in f:
        ls = '/'.join(jieba.cut(line, cut_all=False)).split('/')
        for word in ls:
            if word not in voc.keys():
                voc[word] = 1
            else:
                voc[word] += 1
    f = open('./data/train.dec', 'r', encoding='utf-8')
    for line in f:
        ls = '/'.join(jieba.cut(line, cut_all=False)).split('/')
        for word in ls:
            if word not in voc.keys():
                voc[word] = 1
            else:
                voc[word] += 1
    voc = sorted(voc.items(),key=lambda voc:voc[1],reverse=True)
    voc = voc if len(voc)< max_length else voc[:max_length]
    voc_ls = [x for x,y in voc if y>min_count]
    voc_ls = _START_VOCAB + voc_ls
    for x,y in enumerate(voc_ls):
        voc_final[y] = x
    with open('./data/voc.pkl','wb') as f:
        pickle.dump(voc_final, f, pickle.HIGHEST_PROTOCOL)
    return voc_final
def read_voc_file(filename='./data/voc.pkl'):
    with open(filename,'rb') as f:
        voc = pickle.load(f)
    return voc
def to_id(voc, setence,max_length):
    word_ls = list(setence)
    setence_id =[voc[word] if word in voc.keys() else voc['_UNK'] for word in word_ls]
    setence_id = setence_id + [voc[_EOS]]
    length = len(setence_id)
    if len(setence_id) < max_length:
        setence_id = setence_id + [PAD_ID] * (max_length-len(setence_id))
    return setence_id,length
def get_binary_mask(batch_output):
    binary_masks = []
    for line in batch_output:
        binary_mask = [0 if index == 0 else 1 for index in line]
        binary_masks.append(binary_mask)
    binary_masks_vec = torch.ByteTensor(binary_masks)
    return binary_masks_vec
def get_batch(voc, batch_size, pairs,max_input_length,max_target_length):
    batch_pairs = random.sample(pairs,batch_size)
    batch_pairs = sorted(batch_pairs,key = lambda x:len(x[0]),reverse=True)
    inputs, targets, lengths,target_lengths = [], [], [], []
    for pair in batch_pairs:
        input, length = to_id(voc, pair[0], max_input_length)
        target, target_length = to_id(voc, pair[1], max_target_length)
        inputs.append(input)
        lengths.append(length)
        targets.append(target)
        target_lengths.append(target_length)

    binary_masks_vec = get_binary_mask(targets).transpose(0, 1)
    inputs_vec = torch.LongTensor(inputs).transpose(0, 1)
    lengths_vec = torch.LongTensor(lengths)
    outputs_vec = torch.LongTensor(targets).transpose(0, 1)
    max_target_length = max(target_lengths)
    return inputs_vec, lengths_vec, outputs_vec, binary_masks_vec, max_target_length
def get_weight(voc,pretrained_embedding_path = 'sgns.baidubaike.bigram-char',embedding_dim=300): #
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_embedding_path, binary=False)
    voc_size = len(voc)
    weight = torch.zeros(voc_size,embedding_dim)
    for word in voc.keys():
        index = voc[word]
        if word in model.index2word:
            weight[index, :] = torch.from_numpy(model[word])
    embedding = torch.nn.Embedding.from_pretrained(weight)
    return embedding
def to_word(voc, index):
    return voc.keys[index]
voc = get_directionary(5)
voc= read_voc_file()
#voc = read_voc_file()
print(voc)
#print(get_weight(voc,'/home/zhangjinjie/knowledge-driven-dialogue-master2/knowledge-driven-dialogue-master/generative_pt/sgns.baidubaike.bigram-char'))
#inputs_vec, lengths_vec, outputs_vec, binary_masks_vec, target_lengths= get_batch(voc,3,pairs,150,60)
'''
print(inputs_vec)
print(lengths_vec)
print(outputs_vec)
print(binary_masks_vec)
print(target_lengths)
'''
