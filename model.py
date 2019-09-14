import torch
import torch.nn as nn
import torch.nn.functional as F
Use_Cuda = torch.cuda.is_available()
device = torch.device('cuda' if Use_Cuda else 'cpu')
class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding,  n_layers=1, dropout= 0, embedding_dim=300 ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.embedding.weight.requires_grad = True
        self.embedding_dim =  embedding_dim

        self.gru =nn.GRU(embedding_dim,hidden_size,n_layers,
                         dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq,input_lengths,hidden=None):
        '''

        :param input_seq: shape = [max_length, batch_size],(转置后的结果使得t时刻的batch个数据在内存(显存)中是连续的，从而读取效率更高)
        :param input_lengths: shape = [batch_size] 一批次中每个句子对应的句子长度列表
        :param hidden:  Encoder的初始hidden输入，默认为None
            shape =[num_layers*num_directions,batch_size,hidden_size]
            实际排列顺序是num_directions在前面,
            即对于4层双向的GRU, num_layers*num_directions = 8
            前4层是正向: [:4, batch_size, hidden_size]
            后4层是反向: [4:, batch_size, hidden_size]
        outputs:
            所有时刻的hidden层的输出
            shape: [max_seq_len, batch_size, hidden_size*num_directions]
            正向: [:, :, :hidden_size] 反向: [:, :, hidden_size:]
            最后对双向GRU求和,得到最终的outputs: shape为[max_seq_len, batch_size, hidden_size]
            输出的hidden:[num_layers*num_directions, batch_size, hidden_size]
        hideen:
            最后一个时刻，每一层网络的输出
        '''


        embedded = self.embedding(input_seq)  #shape = [max_length,batch_size,embedding_dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden  =self.gru(packed,hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden



class Attn(nn.Module):
    def __init__(self,attn_method,hidden_size):
        super(Attn,self).__init__()
        self.method =attn_method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size*[0]))
    def dot_score(self, hidden, encoder_outputs):
        '''
        encoder_outputs:
        shape = [max_length, batch_size, num_layers*hidden_size]
        hidden = [1, batch_size, hidden_size]   此时的hidden是Decoder阶段的某一个时刻的hidden，值得注意一下，广播机制

        注意: attention method: 'dot', Hadamard乘法,对应元素相乘，用*就好了

        torch.sum()将第三维相加，最终shape为: [max_lenth, batch_size]
        '''
        return torch.sum(hidden * encoder_outputs, dim=2)#(dim = 0, 1, 2...)是从0开始的

    def general_score(self, hidden, encoder_outputs):   #Luong's multiplicative style
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden , encoder_outputs):   #Bahdanau's multiplicative style
        '''
        由于是双向RNN的原因，encoder的输出hidden_size是的decoder层2倍
        '''
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1),
                                     encoder_outputs),2)).tanh()
        print('self.v')
        print(self.v )
        print('energy.......')
        print(energy)
        print(torch.sum(self.v * energy, dim=2))
        return torch.sum(self.v * energy, dim=2)
    def forward(self, hidden, encoder_outputs):
        '''
        attn_energies:
        shape = [max_length,batch_size]
        '''
        if self.method == 'dot' :
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()#转制为shape = [batch_size, max_length]

        #对dim=1进行softmax,然后插入维度[batch_size, 1, max_seq_len]
        return F.softmax(attn_energies,dim = 1).unsqueeze(1)
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model, embedding, hidden_size, output_size, n_layers=1, dropout = 0.1,embedding_dim=300):
        super(LuongAttnDecoderRNN,self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dim =embedding_dim

        #定义decoder layers
        self.embedding = embedding
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)
        self.attn = Attn(attn_model,hidden_size)
    def forward(self, input_step, last_hidden, encoder_outputs):
        '''

        input_step:
        来自上一个时刻的输出
        shape = [1,batch_size]
        last_hidden:
        初始值为encoder的最后时刻的最后一层hidden输出，传入的是encoder_hidden的正向部分,
            即encoder_hidden[:decoder.num_layers], 为了和decoder对应,所以取的是decoder的num_layers
            shape为[num_layers, batch_size, hidden_size]
        encoder_outputs:
        shape = [max_length, batch_size,hidden_size]
        '''
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        #rnn_output : [1, batch_size, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        rnn_output, hidden = self.gru(embedded,last_hidden)
        #attn_weight 为注意力分布情况,shape = [batch_size, 1, max_seq_len]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        #encoder_output 经过tranpose后的形状是[batch_size, max_lenth, hidden_size]
        # bmm批量矩阵相乘,
        # attn_weights是batch_size个矩阵[1, max_seq_len]
        # encoder_outputs.transpose(0, 1)是batch_size个[max_seq_len, hidden_size]
        # 相乘结果context为: [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # rnn_output从(1, batch_size, hidden_size)变成(batch_size, hidden_size)
        rnn_output = rnn_output.squeeze(0)

        # context从[batch_size, 1, hidden_size]变成[batch_size, hidden_size]
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output,context),1)

        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)

        output = F.softmax(output, dim=1)

        return output, hidden












