import re
import time
import random
import jieba
from tqdm import tqdm
import logging
import math
import os

import torch
import torch.nn as nn
from torch import optim

from data_processing import read_voc_file
from data_processing import get_pairs
from data_processing import get_directionary
from data_processing import get_weight
from data_processing import get_batch
from model import EncoderRNN, LuongAttnDecoderRNN
from data_processing import GO_ID
from Config import Config
use_cuda = torch.cuda.is_available()
#device = 'cuda' if use_cuda else 'cpu'
device = 'cpu'

def maskNLLoss(inp, target, mask):
    '''

    inp:
    shape = [batch_size, voc_length]
    target:[batch_size]
    shape = [batch_size]

    mask:

    '''
    
    nTotal = mask.sum()
    #torch.gather函数改变张量的形状，target.view(-1,1)改变形状为[batch_size, 1]
    #
    crossEntropy = - torch.log(torch.gather(inp , 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss ,nTotal.item()
def train_by_batch(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
                   encoder_optimizer, decoder_optimizer, batch_size, clip,use_teacher_forcing):

    #梯度清0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()


    #给张量分配运算设备
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    #初始化变量
    loss = 0
    print_losses = []
    n_total = 0

    #Encoder 的计算
    encoder_outputs, encoder_hidden  = encoder(input_variable,lengths)
    #Decoder的初始值是SoS,我们需要构早(1,batch_size)的输入，表示第一个时刻batch个输入
    decoder_input = torch.LongTensor([[GO_ID for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    #判断是否使用 teacher forcing
    '''
    teacher forcing :不管模型在t-1时刻做什么预测都把t-1时刻的正确答案作为t时刻的输入。
    但是如果只用teacher forcing也有问题，因为在真实的Decoder的是是没有老师来帮它纠正错误的。
    所以比较好的方法是更加一个teacher_forcing_ratio参数随机的来确定本次训练是否teacher forcing。
    
    '''
    print(encoder_outputs)
    use_teacher_forcing = True if random.random() < use_teacher_forcing else False
    #一次性处理一个时刻
    if use_teacher_forcing:
        
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Teacher forcing:下一个时刻的输入时上一个时刻的正确答案
            decoder_input = target_variable[t].view(1,-1)
            #print('decoder_out......')
            #print(decoder_output)
            #计算累计的loss
            mask_loss, nTotal = maskNLLoss(decoder_output , target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_total += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)    #不使用teacher forcing返回概率最高的词
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            ##计算累计的loss
            mask_loss, nTotal = maskNLLoss(decoder_output , target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_total += nTotal
    #反向计算
    loss.backward()

    #encoder 和 decoder进行梯度裁剪
    _ = nn.utils.clip_grad_norm(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm(decoder.parameters(), clip)

    #更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_total
def train():
    parameter = Config()
    model_name = parameter.model_name
    save_dir = parameter.save_dir
    loadFilename = parameter.model_ckpt

    pretrained_embedding_path = parameter.pretrained_embedding_path
    max_input_length = parameter.max_input_length
    max_generate_length = parameter.max_generate_length
    embedding_dim = parameter.embedding_dim
    batch_size = parameter.batch_size
    hidden_size = parameter.hidden_size
    attn_model = parameter.method
    dropout = parameter.dropout
    clip = parameter.clip
    num_layers = parameter.num_layers

    learning_rate = parameter.learning_rate
    teacher_forcing_ratio = parameter.teacher_forcing_ratio
    decoder_learning_ratio = parameter.decoder_learning_ratio
    n_iteration = parameter.epoch
    print_every = parameter.print_every
    save_every = parameter.save_every
    print(max_input_length,max_generate_length)
    #data
    voc = read_voc_file() #从保存的词汇表之中读取词汇
    print(voc)
    pairs = get_pairs()
    train_batches = None
    try :
        training_batches = torch.load( os.path.join(save_dir, '{}_{}_{}.tar'.format(n_iteration, 'training_batches', batch_size)))
    except FileNotFoundError:
        training_batches = [get_batch(voc, batch_size, pairs, max_input_length, max_generate_length) for _ in
                            range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, '{}_{}_{}.tar'.format(n_iteration, 'training_batches', batch_size)))

    #model
    checkpoint = None
    print('Building encoder and decoder ...')
    if pretrained_embedding_path == None :
        embedding = nn.Embedding(len(voc), embedding_dim)
    else:
        embedding = get_weight(voc, pretrained_embedding_path, embedding_dim)
    print('embedding加载完成')
    encoder = EncoderRNN(hidden_size, embedding, num_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, len(voc), num_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])
    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']
    
    f = open('record.txt','w',encoding ='utf-8')
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        loss = train_by_batch(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size,clip,teacher_forcing_ratio)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', model_name, '{}-{}_{}'.format(num_layers, num_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration,  'backup_bidir_model')))
            print(perplexity)

if __name__ == '__main__':
    train()








