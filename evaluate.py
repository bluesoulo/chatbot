import torch
import jieba
from util.greedySearch import GreedySearchDecoder
from model import EncoderRNN
from model import LuongAttnDecoderRNN
from data_processing import to_id
from data_processing import get_weight
from Config import Config
from data_processing import GO_ID,EOS_ID
from data_processing import read_voc_file
max_length = 150
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

def get_input_line(filename):
    f = open(filename, 'r',encoding='utf-8')
    input = [ '/'.join(jieba.cut(line, cut_all=False)).split('/') for line in f]
    return input
print(len(get_input_line('./test/test.txt')))
def get_batch_id(sentences):
    voc = read_voc_file()
    lengths, input_batches = [], []
    for line in sentences:
        sentence_id, length = to_id(voc, line, max_length)
        lengths.append(length)
        input_batches.append(sentence_id)
    return input_batches, lengths
def generate(input_seq, searcher, sos, eos, device):
    #input_seq: 已分词且转为索引的序列
    #input_batch: shape: [1, seq_len] ==> [seq_len,1] (即batch_size=1)
    input_batch = [input_seq]
    input_lengths = torch.tensor([len(seq) for seq in input_batch])
    input_batch = torch.LongTensor(input_batch).transpose(0,1)
    input_batch = input_batch.to(device)
    input_lengths = input_lengths.to(device)
    tokens, scores = searcher(sos, eos, input_batch, input_lengths, 150, device)
    return tokens,scores
def eval():
    parameter = Config()
    # 加载参数
    save_dir = parameter.save_dir
    loadFilename = parameter.model_ckpt

    pretrained_embedding_path = parameter.pretrained_embedding_path
    dropout = parameter.dropout
    hidden_size = parameter.hidden_size
    num_layers = parameter.num_layers
    attn_model = parameter.method

    max_input_length = parameter.max_input_length
    max_generate_length = parameter.max_generate_length
    embedding_dim = parameter.embedding_dim
    #加载embedding
    voc = read_voc_file('./data/voc.pkl')
    embedding = get_weight(voc,pretrained_embedding_path)
    #输入
    inputs = get_input_line('./test/test.txt')
    input_batches, lengths = get_batch_id(inputs)
    #
    encoder = EncoderRNN(hidden_size, embedding, num_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model,embedding,hidden_size,len(voc),num_layers,dropout)
    if loadFilename == None:
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(loadFilename, map_location=lambda s, l: s)
    print(checkpoint['plt'])
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])
    answer =[]
    with torch.no_grad():
        encoder.to(device)
        decoder.to(device)
        #切换到测试模式
        encoder.eval()
        decoder.eval()
        search = GreedySearchDecoder(encoder, decoder)
        for input_batch in input_batches:
            #print(input_batch)
            token,score = generate(input_batch, search, GO_ID, EOS_ID, device)
            print(token)
            answer.append(token)
        print(answer)
    return answer


if __name__ == '__main__':
    eval()




